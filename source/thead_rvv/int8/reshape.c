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

#include "shl_thead_rvv.h"

/*************************************************************
    note: support flexible vlen
*************************************************************/

int shl_rvv_reshape_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    int size = csinn_tensor_byte_size(input);

    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
        const int vl = vsetvl_e8m1(packn);

        int outer_size = input->dim[1];
        int inner_size = input->dim[2] * input->dim[3];
        for (int ic = 0; ic + packn - 1 < outer_size; ic += packn) {
            int8_t *out_ptr = output_data + ic * inner_size;
            for (int i = 0; i < inner_size; i++) {
                vint8m1_t _input = vle8_v_i8m1(input_data, vl);
                input_data += vl;
                vsse8_v_i8m1(out_ptr, inner_size * sizeof(int8_t), _input, vl);
                out_ptr += 1;
            }
        }
        // XXX: adapt fc ???
    } else if (input->layout == CSINN_LAYOUT_NHWC) {
        const int packn = csrr_vlenb() / sizeof(int8_t);
        int vl = vsetvl_e8m1(packn);

        int outer_size = input->dim[1] * input->dim[2];
        int inner_size = input->dim[3];
        for (int i = 0; i < outer_size; i++) {
            int8_t *out_ptr = output_data + i;
            int size = inner_size;
            while (size > 0) {
                vl = vsetvl_e8m1(size);
                vint8m1_t _input = vle8_v_i8m1(input_data, vl);
                input_data += vl;
                vsse8_v_i8m1(out_ptr, outer_size * sizeof(int8_t), _input, vl);
                out_ptr += vl * outer_size;
                size -= vl;
            }
        }
    } else {
        memcpy(output_data, input_data, size);
    }
    return CSINN_TRUE;
}
