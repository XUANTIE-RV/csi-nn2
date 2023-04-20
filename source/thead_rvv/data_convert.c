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
#ifdef XTHEADVDOT
int shl_rvv_data_convert_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_siso_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    // TODO: corrected output quantization parameters ???
    if (input->dtype == CSINN_DTYPE_INT8 && output->dtype == CSINN_DTYPE_INT4) {
        cb->exec = shl_rvv_data_convert_int8_to_int4;
    } else if (input->dtype == CSINN_DTYPE_INT4 && output->dtype == CSINN_DTYPE_INT8) {
        cb->exec = shl_rvv_data_convert_int4_to_int8;
    }
    return CSINN_TRUE;
}

int shl_rvv_data_convert_int8_to_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_siso_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    int size = csinn_tensor_size(input);
    int size2 = size / 2 * 2;
    while (size2 > 0) {
        int vl = vsetvl_e8m2(size2);
        vint8m2_t _input = vle8_v_i8m2(input_data, vl);
        vint8m2_t _tmp = vssra_vx_i8m2(_input, 4, vl);
        vint8m1_t _res = vpnclip_wx_i8m1(vreinterpret_v_i8m2_i16m2(_tmp), 0, vl / 2);
        vse8_v_i8m1(output_data, _res, vl / 2);
        input_data += vl;
        output_data += vl / 2;
        size2 -= vl;
    }
    if (size & 1) {
        *output_data = (*input_data + 8) >> 4;  // round arithmetic shift right
    }
    return CSINN_TRUE;
}

int shl_rvv_data_convert_int4_to_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_siso_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    int size = csinn_tensor_size(input);
    int size_2 = size / 2;
    while (size_2 > 0) {
        int vl = vsetvl_e8m1(size_2);
        vint8m1_t _input = vle8_v_i8m1(input_data, vl);
        vint16m2_t _tmp = vpwadd_vx_i16m2(_input, 0, vl);
        vint8m2_t _res = vsll_vx_i8m2(vreinterpret_v_i16m2_i8m2(_tmp), 4, vl * 2);
        vse8_v_i8m2(output_data, _res, vl * 2);
        input_data + vl;
        output_data += vl * 2;
        size_2 -= vl;
    }
    if (size & 1) {
        *output_data = (*input_data) << 4;
    }
    return CSINN_TRUE;
}
#endif
