/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#include "csi_nn.h"
#include "csi_utils.h"
#include <math.h>

int csi_psroipooling_f32(struct csi_tensor *data,
                         struct csi_tensor *rois,
                         struct csi_tensor *output,
                         struct psroipooling_params *params)
{
    float *output_data = output->data;
    float *bottom_data = data->data;
    float *bottom_rois = rois->data;

    int width = data->dim[3];
    int height = data->dim[2];
    int num_rois = rois->dim[0];

    for(int n = 0; n < num_rois; n++){
        int   roi_add = n * 5;
        float roi_start_w = (float)(round(bottom_rois[roi_add + 1]) * params->spatial_scale);
        float roi_start_h = (float)(round(bottom_rois[roi_add + 2]) * params->spatial_scale);
        float roi_end_w   = (float)(round(bottom_rois[roi_add + 3] + 1.0) * params->spatial_scale);
        float roi_end_h   = (float)(round(bottom_rois[roi_add + 4] + 1.0) * params->spatial_scale);

        float roi_height = fmaxf(roi_end_h - roi_start_h, 0.1);
        float roi_width  = fmaxf(roi_end_w - roi_start_w, 0.1);
        float bin_size_h = (float)(roi_height) / (float)(params->group_size);
        float bin_size_w = (float)(roi_width) / (float)(params->group_size);

        int ctop, ph, pw, h, w;
        for (ctop = 0; ctop < params->output_dim; ++ctop)
        {
            for (ph = 0; ph < params->group_size; ++ph)
            {
                for (pw = 0; pw < params->group_size; ++pw)
                {
                    int index  = n * params->output_dim * params->group_size * params->group_size + \
                                    ctop * params->group_size * params->group_size + ph * params->group_size + pw;

                    int hstart = (int)(floor((float)(ph)    * bin_size_h + roi_start_h));
                    int wstart = (int)(floor((float)(pw)    * bin_size_w + roi_start_w));
                    int hend   = (int)(ceil((float)(ph + 1) * bin_size_h + roi_start_h));
                    int wend   = (int)(ceil((float)(pw + 1) * bin_size_w + roi_start_w));
                    hstart = fminf(fmaxf(hstart, 0), height);
                    hend   = fminf(fmaxf(hend  , 0), height);
                    wstart = fminf(fmaxf(wstart, 0), width);
                    wend   = fminf(fmaxf(wend  , 0), width);

                    int is_empty = (hend <= hstart) || (wend <= wstart);
                    int gw = pw;
                    int gh = ph;
                    int c = (ctop*params->group_size + gh)*params->group_size + gw;
                    float out_sum = 0;
                    for (h = hstart; h < hend; ++h)
                    {
                        for (w = wstart; w < wend; ++w)
                        {
                            int bottom_index = h * width + w;
                            out_sum += bottom_data[c * height * width + bottom_index];
                        }
                    }
                    float bin_area = (hend - hstart) * (wend - wstart);
                    if (is_empty)
                    {
                        output_data[index] = 0;
                    }else
                    {
                        output_data[index] = out_sum/bin_area;
                    }
                }
            }
        }
    }
    output->data = output_data;
    return CSINN_TRUE;
}

int csi_psroipooling_u8(struct csi_tensor *data,
                        struct csi_tensor *rois,
                        struct csi_tensor *output,
                        struct psroipooling_params *params)
{
    uint8_t *output_data = output->data;


    // init output
    struct csi_tensor float_output;
    memcpy(&float_output, output, sizeof(struct csi_tensor));
    int64_t outer_size = 1;
    for (int i = 0; i < output->dim_count; ++i) {
        outer_size *= output->dim[i];
    }
    float *float_output_data = malloc(outer_size * sizeof(float));
    float_output.data = float_output_data;

    // convert input to float
    struct csi_tensor float_data;
    memcpy(&float_data, data, sizeof(struct csi_tensor));
    int size = 1;
    for (int j = 0; j < data->dim_count; j++){
        size *= data->dim[j];
    }
    float *float_data_item_data = malloc(size * sizeof(float));

    uint8_t *data_item_data = data->data;
    for (int k = 0; k < size; k++) {
        float_data_item_data[k] = csi_dequantize_u8_to_f32(data_item_data[k], data->zero_point,
                                                data->multiplier, data->shift);
    }
    float_data.data = float_data_item_data;

    // convert input to float
    struct csi_tensor float_rois;
    memcpy(&float_rois, rois, sizeof(struct csi_tensor));
    size = 1;
    for (int j = 0; j < rois->dim_count; j++){
        size *= rois->dim[j];
    }
    float *float_rois_item_data = malloc(size * sizeof(float));

    uint8_t *rois_item_data = rois->data;
    for (int k = 0; k < size; k++) {
        float_rois_item_data[k] = csi_dequantize_u8_to_f32(rois_item_data[k], rois->zero_point,
                                                rois->multiplier, rois->shift);
    }
    float_rois.data = float_rois_item_data;

    params->spatial_scale = csi_dequantize_u8_to_f32(1.0, 0, params->spatial_scale_multiplier, params->spatial_scale_shift);

    csi_psroipooling_f32(&float_data, &float_rois, &float_output, params);

    for (int i = 0; i < outer_size; i++) {
        output_data[i] = csi_quantize_f32_to_u8(float_output_data[i], output->zero_point,
                                          output->multiplier, output->shift);
    }
    free(float_data_item_data);
    free(float_rois_item_data);
    free(float_output_data);
    return CSINN_TRUE;
}


int csi_psroipooling_init(struct csi_tensor *data,
                          struct csi_tensor *rois,
                          struct csi_tensor *output,
                          struct psroipooling_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_PSROIPOOLING, data->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_psroipooling(struct csi_tensor *data,
                     struct csi_tensor *rois,
                     struct csi_tensor *output,
                     struct psroipooling_params *params)
{
    if (params->bc != NULL) {
        params->bc(data, rois, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}