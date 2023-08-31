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

#include "rvv/rvv.h"

static void transpose_021_int8(int8_t *src, int8_t *dst, int batch, int inner_size, int outer_size)
{
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < outer_size; i++) {
            int size = inner_size;
            int8_t *d_ptr = dst + i;
            while (size > 0) {
                int vl = vsetvl_e8m4(size);
                vint8m4_t _in = vle8_v_i8m4(src, vl);
                src += vl;
                vsse8_v_i8m4(d_ptr, outer_size * sizeof(int8_t), _in, vl);
                d_ptr += vl * outer_size;
                size -= vl;
            }
        }
        dst += inner_size * outer_size;
    }
}

static int transpose_tail_coincide_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_transpose_params *params, int tail)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int32_t *in_dim = input->dim;
    int32_t *out_dim = output->dim;
    int32_t *permute = params->permute;
    int permute_num = params->permute_num;

    int tail_size = 1;
    for (int i = permute_num - 1; i >= permute_num - tail; i--) {
        tail_size *= in_dim[i];
    }

    int32_t *idx = (int32_t *)shl_mem_alloc(permute_num * sizeof(int32_t));

    int d = 0;
    while (idx[0] < in_dim[0]) {
        if (d == permute_num - tail) {
            int8_t *src = input_data + shl_rvv_transpose_get_in_index(in_dim, idx, permute_num);
            int8_t *dst =
                output_data + shl_rvv_transpose_get_out_index(out_dim, idx, permute, permute_num);
            int i = 0;
            while (i < tail_size) {
                int vl = vsetvl_e8m4(tail_size - i);
                vint8m4_t _in = vle8_v_i8m4(src, vl);
                vse8_v_i8m4(dst, _in, vl);
                src += vl;
                dst += vl;
                i += vl;
            }
            d -= 1;
            idx[d] += 1;
        } else {
            if (idx[d] < in_dim[d]) {
                d += 1;
            } else {
                idx[d] = 0;
                d -= 1;
                idx[d] += 1;
            }
        }
    }

    shl_mem_free(idx);
    return CSINN_TRUE;
}

int shl_rvv_transpose_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_transpose_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input);
    }

    if (params->permute_num == 4 && params->permute[0] == 0 && params->permute[1] == 1 &&
        params->permute[2] == 2 && params->permute[3] == 3) {
        int8_t *input_data = (int8_t *)input->data;
        int8_t *output_data = (int8_t *)output->data;
        int sizeb = csinn_tensor_byte_size(input);
        memcpy(output_data, input_data, sizeb);
        return CSINN_TRUE;
    } else if (params->permute_num == 4 && params->permute[0] == 0 && params->permute[1] == 2 &&
               params->permute[2] == 3 && params->permute[3] == 1) {
        int batch = input->dim[0];
        int inner_size = input->dim[2] * input->dim[3];
        int outer_size = input->dim[1];
        transpose_021_int8(input->data, output->data, batch, inner_size, outer_size);
        return CSINN_TRUE;
    } else if (params->permute_num == 3 && params->permute[0] == 0 && params->permute[1] == 2 &&
               params->permute[2] == 1) {
        int batch = input->dim[0];
        int inner_size = input->dim[2];
        int outer_size = input->dim[1];
        transpose_021_int8(input->data, output->data, batch, inner_size, outer_size);
        return CSINN_TRUE;
    }

    int tail = shl_rvv_transpose_get_tail(params->permute, params->permute_num);
    if (tail > 0) {
        return transpose_tail_coincide_int8(input, output, params, tail);
    }

    return shl_ref_transpose_quant(input, output, params);
}
