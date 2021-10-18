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

/* https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/l2normalization.h */

int csi_l2_normalization_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct l2n_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    const int trailing_dim = input->dim_count - 1;
    int outer_size = 1;
    const int depth = input->dim[trailing_dim];

    for (int i = 0; i < trailing_dim; i++) {
        outer_size *= input->dim[i];
    }

    for (int i = 0; i < outer_size; ++i) {
        float squared_l2_norm = 0;
        for (int c = 0; c < depth; ++c) {
            const float val = input_data[depth * i + c];
            squared_l2_norm += val * val;
        }
        const float l2_norm = sqrt(squared_l2_norm + params->epsilon);
        for (int c = 0; c < depth; ++c) {
            output_data[depth * i + c] = input_data[depth * i + c] / l2_norm;
        }
    }
    return CSINN_TRUE;
}

int csi_l2_normalization_u8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct l2n_params *params)
{
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    const int trailing_dim = input->dim_count - 1;
    int outer_size = 1;
    const int depth = input->dim[trailing_dim];

    for (int i = 0; i < trailing_dim; i++) {
        outer_size *= input->dim[i];
    }

    for (int i = 0; i < outer_size; ++i) {
        float squared_l2_norm = 0;
        for (int c = 0; c < depth; ++c) {
            const float val = csi_dequantize_u8_to_f32(input_data[depth * i + c], input->zero_point,
                                            input->multiplier, input->shift);
            squared_l2_norm += val * val;
        }
        const float l2_norm = sqrt(squared_l2_norm + params->epsilon);
        for (int c = 0; c < depth; ++c) {
            output_data[depth * i + c] = csi_quantize_f32_to_u8(input_data[depth * i + c] / l2_norm, output->zero_point,
                                            output->multiplier, output->shift);
        }
    }
    return CSINN_TRUE;
}

int csi_l2_normalization_init(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct l2n_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_L2N, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_l2_normalization(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct l2n_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}