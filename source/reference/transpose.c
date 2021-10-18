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
#include <assert.h>

static int csi_transpose_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct transpose_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    const int unextended_output_size = output->dim_count;;
    assert(unextended_output_size < 5);
    const int input_ext_size = 4 - input->dim_count;
    const int output_ext_size = 4 - unextended_output_size;

    // The perm data is extended to match the output, each index incremented by
    // the amount of front padding of the input shape.
    int extended_perm[4];
    for (int i = 0; i < output_ext_size; ++i) {
        extended_perm[i] = i;
    }
    for (int i = 0; i < unextended_output_size; ++i) {
        extended_perm[i + output_ext_size] = params->permute[i] + input_ext_size;
    }

    int out_sizes[4];
    // Compute the inverse permutation array so we can do an output centered
    // transpose. Also, check to make sure output_dims is matching input_dims.
    for (int k = 0; k < 4; k++) {
        out_sizes[k] = output->dim[k];
    }

    // Naive transpose loop (iterate on output index and compute input index).
    int o[4]; // loop index (on output).
    int i[4];
    for (o[3] = 0; o[3] < out_sizes[3]; o[3]++) {
        i[extended_perm[3]] = o[3];
        for (o[2] = 0; o[2] < out_sizes[2]; o[2]++) {
            i[extended_perm[2]] = o[2];
            for (o[1] = 0; o[1] < out_sizes[1]; o[1]++) {
                i[extended_perm[1]] = o[1];
                for (o[0] = 0; o[0] < out_sizes[0]; o[0]++) {
                    i[extended_perm[0]] = o[0];
                    output_data[csi_get_index(output->dim, o[0], o[1], o[2], o[3])] =
                        input_data[csi_get_index(input->dim, i[0], i[1], i[2], i[3])];
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_transpose_u8(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct transpose_params *params)
{
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    const int unextended_output_size = output->dim_count;;
    assert(unextended_output_size < 8);

    const int input_ext_size = unextended_output_size - input->dim_count;
    const int output_ext_size = unextended_output_size - unextended_output_size;
    int extended_perm[unextended_output_size];
    for (int i = 0; i < output_ext_size; ++i) {
        extended_perm[i] = i;
    }
    for (int i = 0; i < unextended_output_size; ++i) {
        extended_perm[i + output_ext_size] = params->permute[i] + input_ext_size;
    }
    int out_sizes[unextended_output_size];
    for (int k = 0; k < unextended_output_size; k++) {
        out_sizes[k] = output->dim[k];
    }
    int o[unextended_output_size]; // loop index (on output).
    int i[unextended_output_size];
    if (unextended_output_size == 4){
        // Naive transpose loop (iterate on output index and compute input index).
        for (o[3] = 0; o[3] < out_sizes[3]; o[3]++) {
            i[extended_perm[3]] = o[3];
            for (o[2] = 0; o[2] < out_sizes[2]; o[2]++) {
                i[extended_perm[2]] = o[2];
                for (o[1] = 0; o[1] < out_sizes[1]; o[1]++) {
                    i[extended_perm[1]] = o[1];
                    for (o[0] = 0; o[0] < out_sizes[0]; o[0]++) {
                        i[extended_perm[0]] = o[0];
                        output_data[csi_get_index(output->dim, o[0], o[1], o[2], o[3])] =
                            input_data[csi_get_index(input->dim, i[0], i[1], i[2], i[3])];
                    }
                }
            }
        }
    }
    else if (unextended_output_size == 6){
        // Naive transpose loop (iterate on output index and compute input index).
        for (o[5] = 0; o[5] < out_sizes[5]; o[5]++) {
            i[extended_perm[5]] = o[5];
            for (o[4] = 0; o[4] < out_sizes[4]; o[4]++) {
                i[extended_perm[4]] = o[4];
                for (o[3] = 0; o[3] < out_sizes[3]; o[3]++) {
                    i[extended_perm[3]] = o[3];
                    for (o[2] = 0; o[2] < out_sizes[2]; o[2]++) {
                        i[extended_perm[2]] = o[2];
                        for (o[1] = 0; o[1] < out_sizes[1]; o[1]++) {
                            i[extended_perm[1]] = o[1];
                            for (o[0] = 0; o[0] < out_sizes[0]; o[0]++) {
                                i[extended_perm[0]] = o[0];
                                output_data[csi_get_index_6(output->dim, o[0], o[1], o[2], o[3], o[4], o[5])] =
                                    input_data[csi_get_index_6(input->dim, i[0], i[1], i[2], i[3], i[4], i[5])];
                            }
                        }
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

int csi_transpose_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct transpose_params *params)
{
    if (input->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_transpose_u8;
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        params->bc = csi_transpose_f32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_transpose(struct csi_tensor *input,
             struct csi_tensor *output,
             struct transpose_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}

