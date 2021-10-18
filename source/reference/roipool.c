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


// https://github.com/pytorch/pytorch/blob/master/caffe2/operators/roi_pool_op.cc
// defalut input layout: NCHW
int csi_roipool_f32(struct csi_tensor *data,
                    struct csi_tensor *rois,
                    struct csi_tensor *output,
                    struct roi_pool_params *params)
{
    float *output_data = (float *)output->data;
    float *bottom_data = (float *)data->data;
    float *bottom_rois = (float *)rois->data;

    int batch = data->dim[0];
    int channel = data->dim[1];
    int height = data->dim[2];
    int width  = data->dim[3];
    int num_rois = rois->dim[0];

    int pooled_height = params->pooled_size_h;
    int pooled_width  = params->pooled_size_w;

    for(int n = 0; n < num_rois; n++) {
        int roi_add = n * 5;
        int roi_batch_idx = bottom_rois[roi_add];
        assert(roi_batch_idx < batch);
        float roi_start_w = (float)(round(bottom_rois[roi_add + 1]) * params->spatial_scale);
        float roi_start_h = (float)(round(bottom_rois[roi_add + 2]) * params->spatial_scale);
        float roi_end_w   = (float)(round(bottom_rois[roi_add + 3]) * params->spatial_scale);
        float roi_end_h   = (float)(round(bottom_rois[roi_add + 4]) * params->spatial_scale);

        float roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);
        float roi_width  = fmaxf(roi_end_w - roi_start_w + 1, 1);
        float bin_size_h = (float)(roi_height) / (float)(pooled_height);
        float bin_size_w = (float)(roi_width) / (float)(pooled_width);

        const float *batch_data = bottom_data + roi_batch_idx * channel * height * width;

        for (int c = 0; c < channel; ++c) {
            for (int ph = 0; ph < pooled_height; ++ph) {
                for (int pw = 0; pw < pooled_width; ++pw) {
                    // Compute pooling region for this output unit:
                    // start (included) = floor(ph * roi_height / pooled_height_)
                    // end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
                    int hstart = (int)(floor((float)(ph)    * bin_size_h + roi_start_h));
                    int wstart = (int)(floor((float)(pw)    * bin_size_w + roi_start_w));
                    int hend   = (int)(ceil((float)(ph + 1) * bin_size_h + roi_start_h));
                    int wend   = (int)(ceil((float)(pw + 1) * bin_size_w + roi_start_w));
                    hstart = fminf(fmaxf(hstart, 0), height);
                    hend   = fminf(fmaxf(hend  , 0), height);
                    wstart = fminf(fmaxf(wstart, 0), width);
                    wend   = fminf(fmaxf(wend  , 0), width);

                    const int pool_index = ph * pooled_width + pw;
                    int is_empty = (hend <= hstart) || (wend <= wstart);

                    *(output_data + pool_index) = is_empty ? 0 : -FLT_MAX;

                    for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                            int index = h * width + w;
                            if(*(batch_data + index) > *(output_data + pool_index)) {
                                *(output_data + pool_index) = *(output_data + pool_index);
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

int csi_roipool_u8(struct csi_tensor *data,
                   struct csi_tensor *rois,
                   struct csi_tensor *output,
                   struct roi_pool_params *params)
{
    uint8_t *output_data = (uint8_t *)output->data;

    // init output
    struct csi_tensor float_output;
    memcpy(&float_output, output, sizeof(struct csi_tensor));
    int64_t out_size = 1;
    for(int i = 0; i < output->dim_count; i++) {
        out_size *= output->dim[i];
    }
    float *float_output_data = (float *)malloc(out_size * sizeof(float));
    float_output.data = float_output_data;

    // convert input(data) to float
    struct csi_tensor float_data;
    memcpy(&float_data, data, sizeof(struct csi_tensor));
    int64_t in_size = 1;
    for(int i = 0; i < data->dim_count; i++) {
        in_size *= data->dim[i];
    }
    float *float_data_item_data = (float *)malloc(in_size * sizeof(float));

    uint8_t *data_item_data = (uint8_t *)data->data;
    for(int i = 0; i < in_size; i++) {
        float_data_item_data[i] = csi_dequantize_u8_to_f32(data_item_data[i], data->zero_point, data->multiplier, data->shift);
    }
    float_data.data = float_data_item_data;

    // convert input(rois) to float
    struct csi_tensor float_rois;
    memcpy(&float_rois, rois, sizeof(struct csi_tensor));
    int64_t rois_size = 1;
    for(int i = 0; i < rois->dim_count; i++) {
        rois_size *= rois->dim[i];
    }
    float *float_rois_item_data = (float *)malloc(rois_size * sizeof(float));

    uint8_t *rois_item_data = (uint8_t *)rois->data;
    for(int i = 0; i < rois_size; i++) {
        float_rois_item_data[i] = csi_dequantize_u8_to_f32(rois_item_data[i], rois->zero_point, rois->multiplier, rois->shift);
    }
    float_rois.data = float_rois_item_data;

    // convert params to float
    params->spatial_scale = csi_dequantize_u8_to_f32(1.0, 0, params->spatial_scale_multiplier, params->spatial_scale_shift);

    csi_roipool_f32(&float_data, &float_rois, &float_output, params);

    for(int i = 0; i < out_size; i++) {
        output_data[i] = csi_quantize_f32_to_u8(float_output_data[i], output->zero_point, output->multiplier, output->shift);
    }

    free(float_data_item_data);
    free(float_rois_item_data);
    free(float_output_data);
    return CSINN_TRUE;
}

int csi_roipool_init(struct csi_tensor *data,
                     struct csi_tensor *rois,
                     struct csi_tensor *output,
                     struct roi_pool_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_ROIPOOL, data->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_roipool(struct csi_tensor *data,
                struct csi_tensor *rois,
                struct csi_tensor *output,
                struct roi_pool_params *params)
{
    if (params->bc != NULL) {
        params->bc(data, rois, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}