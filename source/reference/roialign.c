/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* SHL version 2.1.x */

#include "shl_ref.h"

// https://github.com/AceCoooool/RoIAlign-RoIPool-pytorch/blob/master/roialign/roi_align_cpu.cpp

struct PreCalc {
    // left_top, right_top, left_bottom, right_bottom
    int pos1, pos2, pos3, pos4;
    float w1, w2, w3, w4;
};

static void pre_calc_for_bilinear(const int h, const int w, const int pool_h, const int pool_w,
                                  int b_grid_h, int b_grid_w, float start_y, float start_x,
                                  float b_size_h, float b_size_w, struct PreCalc *pre_calc)
{
    int idx = 0;
    for (int ph = 0; ph < pool_h; ++ph) {
        for (int pw = 0; pw < pool_w; ++pw) {
            for (int iy = 0; iy < b_grid_h; ++iy) {
                float yy =
                    start_y + ph * b_size_h + (float)(iy + 0.5f) * b_size_h / (float)(b_grid_h);
                for (int ix = 0; ix < b_grid_w; ++ix) {
                    float xx =
                        start_x + pw * b_size_w + (float)(ix + 0.5f) * b_size_w / (float)(b_grid_w);
                    float x = xx, y = yy;
                    // situation 1: out of range
                    if (y < -1.0 || y > h || x < -1.0 || x > w) {
                        struct PreCalc cc = {0, 0, 0, 0, 0, 0, 0, 0};
                        pre_calc[idx] = cc;
                        idx += 1;
                        continue;
                    }
                    // not exceed 1.0
                    y = y <= 0 ? 0 : (y >= h - 1 ? h - 1 : y);
                    x = x <= 0 ? 0 : (x >= w - 1 ? w - 1 : x);
                    int y_low = (int)y;
                    int x_low = (int)x;
                    int y_high = y_low >= h - 1 ? y_low : y_low + 1;
                    int x_high = x_low >= w - 1 ? x_low : x_low + 1;
                    float ly = y - y_low, lx = x - x_low;
                    float hy = 1.0 - ly, hx = 1.0 - lx;
                    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                    // in the feature map's position and correspond weights
                    struct PreCalc pc;
                    pc.pos1 = y_low * w + x_low;
                    pc.pos2 = y_low * w + x_high;
                    pc.pos3 = y_high * w + x_low;
                    pc.pos4 = y_high * w + x_high;
                    pc.w1 = w1, pc.w2 = w2, pc.w3 = w3, pc.w4 = w4;
                    pre_calc[idx] = pc;
                    idx += 1;
                }
            }
        }
    }
}

int shl_ref_roi_align_f32(struct csinn_tensor *data, struct csinn_tensor *rois,
                          struct csinn_tensor *output, struct csinn_roi_align_params *params)
{
    float *bottom_rois = (float *)rois->data;
    float *input_data = (float *)data->data;
    float *output_data = (float *)output->data;

    int channel = data->dim[1];
    int h = data->dim[2];
    int w = data->dim[3];

    int n_rois = rois->dim[0];
    int pool_h = params->pooled_size_h;  // output->dim[2]
    int pool_w = params->pooled_size_w;  // output->dim[3]
    int ratio = params->sample_ratio;

    for (int n = 0; n < n_rois; ++n) {
        int idx_n = n * channel * pool_h * pool_w;
        int roi_add = n * 5;
        int roi_batch_idx = bottom_rois[roi_add];
        float start_x = bottom_rois[roi_add + 1] * params->spatial_scale;
        float start_y = bottom_rois[roi_add + 2] * params->spatial_scale;
        float end_x = bottom_rois[roi_add + 3] * params->spatial_scale;
        float end_y = bottom_rois[roi_add + 4] * params->spatial_scale;
        // Force malformed ROIs to be 1x1
        float roi_w = fmax(end_x - start_x, 1.0f);
        float roi_h = fmax(end_y - start_y, 1.0f);
        float bin_size_w = roi_w / (pool_w);
        float bin_size_h = roi_h / (pool_h);

        // We use roi_bin_grid to sample the grid and mimic integral
        int bin_grid_h = (ratio > 0) ? ratio : ceil(roi_h / pool_h);
        int bin_grid_w = (ratio > 0) ? ratio : ceil(roi_w / pool_w);
        // We do average (integral) pooling inside a bin
        int count = bin_grid_h * bin_grid_w;
        // get each bin's corresponding position and weights
        struct PreCalc pre_calc[count * pool_h * pool_w];
        pre_calc_for_bilinear(h, w, pool_h, pool_w, bin_grid_h, bin_grid_w, start_y, start_x,
                              bin_size_h, bin_size_w, pre_calc);

        // map to feature map
        for (int c = 0; c < channel; ++c) {
            int idx_nc = idx_n + c * pool_w * pool_h;
            float *offset_feat = input_data + (roi_batch_idx * channel + c) * h * w;
            int pre_calc_idx = 0;
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int idx = idx_nc + ph * pool_w + pw;
                    float output_val = 0.0;
                    for (int iy = 0; iy < bin_grid_h; ++iy) {
                        for (int ix = 0; ix < bin_grid_w; ++ix) {
                            struct PreCalc pc = pre_calc[pre_calc_idx];
                            output_val +=
                                pc.w1 * offset_feat[pc.pos1] + pc.w2 * offset_feat[pc.pos2] +
                                pc.w3 * offset_feat[pc.pos3] + pc.w4 * offset_feat[pc.pos4];
                            pre_calc_idx += 1;
                        }
                    }
                    output_val /= count;
                    output_data[idx] = output_val;
                }
            }
        }
    }
    return CSINN_TRUE;
}
