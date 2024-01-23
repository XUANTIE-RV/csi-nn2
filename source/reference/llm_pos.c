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

#include "reference/ref.h"

int shl_ref_llm_pos_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_llm_pos_params *params)
{
    float *output_data = output->data;
    float *input_data = input->data;

    int batch = params->bsz;
    int seqlen = params->seqlen;
    int start_pos = params->pos[0];
    int inner_size = input->dim[2] * input->dim[3];
    if (params->mode == CSINN_LLM_POS_CACHE_COPY_IN) {
        for (int i = 0; i < batch; i++) {
            int output_index = i * output->dim[1] * inner_size + start_pos * inner_size;
            int input_index = i * input->dim[1] * inner_size;
            int cpy_size = seqlen * inner_size * sizeof(float);

            output_data = params->cache_buffer;
            memcpy(output_data + output_index, input_data + input_index, cpy_size);
        }
    } else if (params->mode == CSINN_LLM_POS_CACHE_COPY_OUT) {
        for (int i = 0; i < batch; i++) {
            int output_index = i * output->dim[1] * inner_size;
            int input_index = i * input->dim[1] * inner_size;
            int cpy_size = (start_pos + seqlen) * inner_size * sizeof(float);
            input_data = params->cache_buffer;

            memcpy(output_data + output_index, input_data + input_index, cpy_size);
        }
    } else if (params->mode == CSINN_LLM_POS_MASK) {
        memcpy(output->data, input->data, csinn_tensor_byte_size(output));
        for (int i = 0; i < input->dim[0] * input->dim[1]; i++) {
            for (int j = 0; j < seqlen; ++j) {
                int32_t *pos = params->pos;
                for (int k = pos[j] + 1; k < seqlen; k++) {
                    int output_index = i * seqlen * seqlen + j * seqlen + k;
                    output_data[output_index] = -INFINITY;
                }
            }
        }
    } else {
        shl_debug_error("Unsupport mode in %s\n", __func__);
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}

int shl_ref_llm_pos_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_llm_pos_params *params)
{
    return shl_ref_llm_pos_f32(input, output, params);
}
