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

#include "shl_c906.h"

int shl_c906_transpose_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_transpose_params *params)
{
    if (params->permute_num == 4 && params->permute[0] == 0 && params->permute[1] == 2 &&
        params->permute[2] == 1 && params->permute[3] == 3) {
        __fp16 *p_input = input->data;
        __fp16 *p_output = output->data;
        int *shape = output->dim;
        int batch = shape[2];
        int shape3 = shape[3];
        int flatten_shape = shape[1] * shape[3];

        if (flatten_shape % 16 == 0) {
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < flatten_shape; j += 16) {
                    int out_pos = i * shape3 + j % shape3 + batch * shape3 * (j / shape3);
                    vfloat16m2_t _output_from_buffer;
                    _output_from_buffer = vle16_v_f16m2(p_input + i * flatten_shape + j, 16);
                    vse16_v_f16m2(p_output + out_pos, _output_from_buffer, 16);
                }
            }

        } else {
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < flatten_shape; j++) {
                    int out_pos = i * shape3 + j % shape3 + batch * shape3 * (j / shape3);
                    p_output[out_pos] = p_input[i * flatten_shape + j];
                }
            }
        }
        return CSINN_TRUE;
    }
    if (params->permute_num == 3 && params->permute[0] == 0 && params->permute[1] == 2 &&
        params->permute[2] == 1) {
        int *shape = output->dim;
        __fp16 *p_input = input->data;
        __fp16 *p_output = output->data;
        for (int i = 0; i < shape[2]; i++)  // 256
        {
            int j = 0;
            for (; j + 15 < shape[1]; j += 16)  // 6
            {
                int out_pos = j * shape[2] + i;
                vfloat16m2_t _output_from_buffer;
                _output_from_buffer = vle16_v_f16m2(p_input + i * shape[1] + j, 16);
                vsse16_v_f16m2(p_output + out_pos, 2 * shape[2], _output_from_buffer, 16);
            }
            if (j != shape[1]) {
                int vl = shape[1] - j;
                int out_pos = j * shape[2] + i;
                vfloat16m2_t _output_from_buffer;
                _output_from_buffer = vle16_v_f16m2(p_input + i * shape[1] + j, vl);
                vsse16_v_f16m2(p_output + out_pos, 2 * shape[2], _output_from_buffer, vl);
            }
        }
        return CSINN_TRUE;
    }
    return shl_ref_siso_callback_base(input, output, params, shl_ref_transpose);
}
