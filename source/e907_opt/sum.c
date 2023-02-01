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

#include "shl_e907.h"

int shl_e907_sum_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_reduce_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    // TODO: move to init api
    float real_scale = input->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale, &output->qinfo->multiplier, &output->qinfo->shift);

    int16_t z1 = input->qinfo->zero_point;
    int16_t z2 = output->qinfo->zero_point;
    int32_t multiplier = output->qinfo->multiplier;
    int32_t shift = output->qinfo->shift;
    intXLEN_t multiplier_32xn = shl_rvp_int32_to_xlen(multiplier);
    intXLEN_t z2_16xn = shl_rvp_int16_to_xlen(z2);

    if (*(params->axis) == -1) {
        int size = 1;
        for (int i = 0; i < input->dim_count; i++) {
            size = size * input->dim[i];
        }
        float res = 0;
        for (int j = 0; j < size; j++) {
            float temp = (input_data[j] - input->qinfo->zero_point) * input->qinfo->scale;
            res = res + temp;
        }
        float ret = round(res / output->qinfo->scale) + output->qinfo->zero_point;
        if (ret > 127)
            ret = 127;
        else if (ret < -128)
            ret = -128;
        *output_data = (int8_t)ret;
    } else {
        int32_t inner_size = 1;
        int32_t outer_size = 1;
        for (int32_t k = 0; k < params->n; k++) {
            outer_size *= params->out_extents[k];
        }
        for (int32_t k = 0; k < params->m; k++) {
            inner_size *= params->inner_extents[k];
        }

        int16_t z1xn = inner_size * z1;
        intXLEN_t z1_16xn = shl_rvp_int16_to_xlen(z1xn);
        const int xlenh = shl_rvp_get_xlenb() >> 1;

        int i = 0;
        for (; i + xlenh - 1 < outer_size; i += xlenh) {
            int8_t *input_ptr = input_data + i;
            intXLEN_t acc = 0;
            for (int j = 0; j < inner_size; j++) {
                int16_t tmp_i16[xlenh];
                shl_rvp_int8_to_int16((input_ptr + j * outer_size), tmp_i16, xlenh);
                intXLEN_t *x_16xn = (intXLEN_t *)(tmp_i16);
                acc = __rv__add16(acc, x_16xn[0]);
            }
            acc = __rv__sub16(acc, z1_16xn);  // - (z1 * inner_size)
            int32_t tmp_i32[xlenh];
            shl_rvp_int16_to_int32((int16_t *)(&acc), tmp_i32, xlenh);
            intXLEN_t *x_32xn = (intXLEN_t *)(tmp_i32);

#if __riscv_xlen == 64
            if (shift < 0) {
                x_32xn[0] = __rv__sra32_u(x_32xn[0], -shift - 1);
                x_32xn[1] = __rv__sra32_u(x_32xn[1], -shift - 1);
            } else {
                x_32xn[0] = __rv__sll32(x_32xn[0], shift + 1);
                x_32xn[1] = __rv__sll32(x_32xn[1], shift + 1);
            }
#elif __riscv_xlen == 32
            if (shift < 0) {
                x_32xn[0] >>= -shift - 1;
                x_32xn[1] >>= -shift - 1;
            } else {
                x_32xn[0] <<= shift + 1;
                x_32xn[1] <<= shift + 1;
            }
#endif

            intXLEN_t mulh[2];
            for (int k = 0; k < 2; k++) {
                mulh[k] = __rv__smmul_u(x_32xn[k], multiplier_32xn);
                mulh[k] = __rv__sclip32(mulh[k], 15);  // narrow 32->16
            }

            int16_t tmp_i16[xlenh];
            shl_rvp_int32_to_int16((int32_t *)mulh, tmp_i16, xlenh);

            intXLEN_t *x_16xn = (intXLEN_t *)tmp_i16;
            for (int j = 0; j < 2; j++) {
                x_16xn[j] = __rv__add16(x_16xn[j], z2_16xn);  // + z2
                x_16xn[j] = __rv__sclip16(x_16xn[j], 7);      // narrow 16->8
            }
            shl_rvp_int16_to_int8((int16_t *)x_16xn, output_data + i, xlenh);
        }

        for (; i < outer_size; i++) {
            int8_t *input_ptr = input_data + i;
            int32_t res = 0;
            for (int j = 0; j < inner_size; j++) {
                int32_t in = (int32_t)input_ptr[j * outer_size];
                res += in;
            }
            res -= z1xn;
            if (shift < 0) {
                res >>= -shift - 1;
            } else {
                res <<= shift + 1;
            }
            res = __rv__smmul_u(res, multiplier);
            res += z2;
            output_data[i] = (int8_t)res;
        }
    }

    return CSINN_TRUE;
}
