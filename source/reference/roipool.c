/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#include <math.h>

#include "shl_ref.h"

// https://github.com/pytorch/pytorch/blob/master/caffe2/operators/roi_pool_op.cc
// defalut input layout: NCHW
int shl_ref_roipool_f32(struct csinn_tensor *data, struct csinn_tensor *rois,
                        struct csinn_tensor *output, struct csinn_roi_pool_params *params)
{
    float *output_data = (float *)output->data;
    float *bottom_data = (float *)data->data;
    float *bottom_rois = (float *)rois->data;

    int batch = data->dim[0];
    int channel = data->dim[1];
    int height = data->dim[2];
    int width = data->dim[3];
    int num_rois = rois->dim[0];

    int pooled_height = params->pooled_size_h;
    int pooled_width = params->pooled_size_w;

    for (int n = 0; n < num_rois; n++) {
        int roi_add = n * 5;
        int roi_batch_idx = bottom_rois[roi_add];
        assert(roi_batch_idx < batch);
        float roi_start_w = (float)round(bottom_rois[roi_add + 1] * params->spatial_scale);
        float roi_start_h = (float)round(bottom_rois[roi_add + 2] * params->spatial_scale);
        float roi_end_w = (float)round(bottom_rois[roi_add + 3] * params->spatial_scale);
        float roi_end_h = (float)round(bottom_rois[roi_add + 4] * params->spatial_scale);

        float roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);
        float roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1);
        float bin_size_h = (float)(roi_height) / (float)(pooled_height);
        float bin_size_w = (float)(roi_width) / (float)(pooled_width);

        const float *batch_data = bottom_data + roi_batch_idx * channel * height * width;

        for (int c = 0; c < channel; ++c) {
            for (int ph = 0; ph < pooled_height; ++ph) {
                for (int pw = 0; pw < pooled_width; ++pw) {
                    // Compute pooling region for this output unit:
                    // start (included) = floor(ph * roi_height / pooled_height_)
                    // end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
                    int hstart = (int)(floor((float)(ph)*bin_size_h + roi_start_h));
                    int wstart = (int)(floor((float)(pw)*bin_size_w + roi_start_w));
                    int hend = (int)(ceil((float)(ph + 1) * bin_size_h + roi_start_h));
                    int wend = (int)(ceil((float)(pw + 1) * bin_size_w + roi_start_w));
                    hstart = fminf(fmaxf(hstart, 0), height);
                    hend = fminf(fmaxf(hend, 0), height);
                    wstart = fminf(fmaxf(wstart, 0), width);
                    wend = fminf(fmaxf(wend, 0), width);

                    const int pool_index = ph * pooled_width + pw;
                    int is_empty = (hend <= hstart) || (wend <= wstart);

                    *(output_data + pool_index) = is_empty ? 0 : -FLT_MAX;

                    for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                            int index = h * width + w;
                            if (*(batch_data + index) > *(output_data + pool_index)) {
                                *(output_data + pool_index) = *(batch_data + index);
                            }
                        }
                    }
                }
            }
            // Increment all data pointers by one channel
            batch_data += height * width;
            output_data += pooled_height * pooled_width;
        }
    }
    return CSINN_TRUE;
}

int shl_ref_roipool_quant(struct csinn_tensor *data, struct csinn_tensor *rois,
                          struct csinn_tensor *output, struct csinn_roi_pool_params *params)
{
    int ret;
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(data);
    struct csinn_tensor *frois = shl_ref_tensor_transform_f32(rois);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    ret = shl_ref_roipool_f32(finput, frois, foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(frois);
    shl_ref_tensor_transform_free_f32(foutput);
    return ret;
}
