/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.8.x */

#include "csi_ref.h"
#include "csi_utils.h"

/* https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/l2normalization.h */

int csi_ref_l2_normalization_f32(struct csi_tensor *input,
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

int csi_ref_l2_normalization_quant(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct l2n_params *params)
{
    return csi_ref_siso_callback_base(input, output, params, csi_ref_l2_normalization_f32);
}
