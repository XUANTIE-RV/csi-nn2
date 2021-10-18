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

static int csi_mean_stride_f32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct reduce_params *params)
{

    float *input_data = input->data;
    float *output_data = output->data;

    int32_t inner_size = 1;
    int32_t out_size = 1;

    for (int32_t k = 0; k < params->n; k++)
    {
        out_size *= params->out_extents[k];
    }

    for (int32_t k = 0; k < params->m; k++)
    {
        inner_size *= params->inner_extents[k];
    }

    for (int32_t out = 0; out < out_size; out++)
    {

        float result = 0;
        int32_t out_index = get_reduction_index(out, params->out_strides, params->out_extents, params->n);
        for (int32_t inner = 0; inner < inner_size; inner++)
        {
            int32_t index = out_index + get_reduction_index(inner, params->inner_strides,
                                                            params->inner_extents, params->m);
            float val = input_data[index];
            result += val;
        }

        output_data[out] = result / inner_size;
    }
    return CSINN_TRUE;
}

static int csi_mean_stride_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params)
{

    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;

    int32_t inner_size = 1;
    int32_t out_size = 1;

    for (int32_t k = 0; k < params->n; k++)
    {
        out_size *= params->out_extents[k];
    }

    for (int32_t k = 0; k < params->m; k++)
    {
        inner_size *= params->inner_extents[k];
    }

    for (int32_t out = 0; out < out_size; out++)
    {

        float result = 0;
        int32_t out_index = get_reduction_index(out, params->out_strides, params->out_extents, params->n);
        for (int32_t inner = 0; inner < inner_size; inner++)
        {
            int32_t index = out_index + get_reduction_index(inner, params->inner_strides,
                                                            params->inner_extents, params->m);
            float val = csi_dequantize_f32(input_data[index], input->offset,
                                           input->multiplier, input->shift);
            result += val;
        }

        output_data[out] = csi_quantize_f32(result / inner_size, output->offset,
                                            output->multiplier, output->shift);
    }
    return CSINN_TRUE;
}

static int csi_mean_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params)
{
    if (params->axis_count != 2 || params->axis[0] != 2 || params->axis[1] != 3 ||
        input->dim_count != 4 || output->dim_count != 4) {
        assert(0);
    }
    struct pool_params pparams;
    pparams.layout = CSINN_NCHW;
    csi_global_averagepool_init(input, output, &pparams);
    csi_global_averagepool(input, output, &pparams);
    return CSINN_TRUE;
}

int csi_mean_init(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct reduce_params *params)
{
    if (params->n == 0 && params->m == 0) {
        if (params->layout == CSINN_NCHW) {
            if (input->dtype == CSINN_DTYPE_UINT8) {
                params->bc = csi_mean_u8;
            } else {
                return CSINN_UNSUPPORT_DTYPE;
            }
        } else {
            return CSINN_UNSUPPORT_LAYOUT;
        }
    } else {
        if (input->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_mean_stride_u8;
        } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->bc = csi_mean_stride_f32;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    }
    return CSINN_TRUE;
}

int csi_mean(struct csi_tensor *input,
             struct csi_tensor *output,
             struct reduce_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}

