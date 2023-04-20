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

/* SHL version 2.1.x */

#include <math.h>

#include "shl_ref.h"

int shl_ref_psroipooling_f32(struct csinn_tensor *data, struct csinn_tensor *rois,
                             struct csinn_tensor *output, struct csinn_psroipooling_params *params)
{
    float *output_data = output->data;
    float *bottom_data = data->data;
    float *bottom_rois = rois->data;

    int width = data->dim[3];
    int height = data->dim[2];
    int num_rois = rois->dim[0];

    for (int n = 0; n < num_rois; n++) {
        int roi_add = n * 5;
        float roi_start_w = (float)(round(bottom_rois[roi_add + 1]) * params->spatial_scale);
        float roi_start_h = (float)(round(bottom_rois[roi_add + 2]) * params->spatial_scale);
        float roi_end_w = (float)(round(bottom_rois[roi_add + 3] + 1.0) * params->spatial_scale);
        float roi_end_h = (float)(round(bottom_rois[roi_add + 4] + 1.0) * params->spatial_scale);

        float roi_height = fmaxf(roi_end_h - roi_start_h, 0.1);
        float roi_width = fmaxf(roi_end_w - roi_start_w, 0.1);
        float bin_size_h = (float)(roi_height) / (float)(params->group_size);
        float bin_size_w = (float)(roi_width) / (float)(params->group_size);

        int ctop, ph, pw, h, w;
        for (ctop = 0; ctop < params->output_dim; ++ctop) {
            for (ph = 0; ph < params->group_size; ++ph) {
                for (pw = 0; pw < params->group_size; ++pw) {
                    int index = n * params->output_dim * params->group_size * params->group_size +
                                ctop * params->group_size * params->group_size +
                                ph * params->group_size + pw;

                    int hstart = (int)(floor((float)(ph)*bin_size_h + roi_start_h));
                    int wstart = (int)(floor((float)(pw)*bin_size_w + roi_start_w));
                    int hend = (int)(ceil((float)(ph + 1) * bin_size_h + roi_start_h));
                    int wend = (int)(ceil((float)(pw + 1) * bin_size_w + roi_start_w));
                    hstart = fminf(fmaxf(hstart, 0), height);
                    hend = fminf(fmaxf(hend, 0), height);
                    wstart = fminf(fmaxf(wstart, 0), width);
                    wend = fminf(fmaxf(wend, 0), width);

                    int is_empty = (hend <= hstart) || (wend <= wstart);
                    int gw = pw;
                    int gh = ph;
                    int c = (ctop * params->group_size + gh) * params->group_size + gw;
                    float out_sum = 0;
                    for (h = hstart; h < hend; ++h) {
                        for (w = wstart; w < wend; ++w) {
                            int bottom_index = h * width + w;
                            out_sum += bottom_data[c * height * width + bottom_index];
                        }
                    }
                    float bin_area = (hend - hstart) * (wend - wstart);
                    if (is_empty) {
                        output_data[index] = 0;
                    } else {
                        output_data[index] = out_sum / bin_area;
                    }
                }
            }
        }
    }
    output->data = output_data;
    return CSINN_TRUE;
}

int shl_ref_psroipooling_quant(struct csinn_tensor *data, struct csinn_tensor *rois,
                               struct csinn_tensor *output,
                               struct csinn_psroipooling_params *params)
{
    int ret;
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(data);
    struct csinn_tensor *frois = shl_ref_tensor_transform_f32(rois);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    ret = shl_ref_psroipooling_f32(finput, frois, foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(frois);
    shl_ref_tensor_transform_free_f32(foutput);
    return ret;
}
