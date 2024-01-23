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

#define a1 0.0705230784
#define a2 0.0422820123
#define a3 0.0092705272
#define a4 0.0001520143
#define a5 0.0002765672
#define a6 0.0000430638

static inline vfloat32m4_t vfpow16_v_f32m4(vfloat32m4_t _x, int vl)
{
    vfloat32m4_t _x2 = vfmul_vv_f32m4(_x, _x, vl);
    vfloat32m4_t _x4 = vfmul_vv_f32m4(_x2, _x2, vl);
    vfloat32m4_t _x8 = vfmul_vv_f32m4(_x4, _x4, vl);
    vfloat32m4_t _x16 = vfmul_vv_f32m4(_x8, _x8, vl);
    return _x16;
}

/*************************************************************************************
 * erf(x) = 1 - 1 / (1 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 + a6*x^6)^16
 **************************************************************************************/
int shl_rvv_erf_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int size = csinn_tensor_size(input);

    while (size > 0) {
        int vl = vsetvl_e32m4(size);
        vfloat32m4_t _x = vle32_v_f32m4(input_data, vl);
        input_data += vl;

        vbool8_t _mask = vmflt_vf_f32m4_b8(_x, 0.0f, vl);
        _x = vfmul_vf_f32m4_m(_mask, _x, _x, -1.0f, vl);

        vfloat32m4_t _x2 = vfmul_vv_f32m4(_x, _x, vl);
        vfloat32m4_t _x3 = vfmul_vv_f32m4(_x2, _x, vl);
        vfloat32m4_t _x4 = vfmul_vv_f32m4(_x2, _x2, vl);
        vfloat32m4_t _x5 = vfmul_vv_f32m4(_x3, _x2, vl);
        vfloat32m4_t _x6 = vfmul_vv_f32m4(_x3, _x3, vl);
        _x = vfmul_vf_f32m4(_x, a1, vl);
        _x2 = vfmul_vf_f32m4(_x2, a2, vl);
        _x3 = vfmul_vf_f32m4(_x3, a3, vl);
        _x4 = vfmul_vf_f32m4(_x4, a4, vl);
        _x5 = vfmul_vf_f32m4(_x5, a5, vl);
        _x6 = vfmul_vf_f32m4(_x6, a6, vl);

        vfloat32m4_t _t = vfmv_v_f_f32m4(1.0f, vl);
        _t = vfadd_vv_f32m4(_t, _x, vl);
        _t = vfadd_vv_f32m4(_t, _x2, vl);
        _t = vfadd_vv_f32m4(_t, _x3, vl);
        _t = vfadd_vv_f32m4(_t, _x4, vl);
        _t = vfadd_vv_f32m4(_t, _x5, vl);
        _t = vfadd_vv_f32m4(_t, _x6, vl);

        vfloat32m4_t _pow = vfpow16_v_f32m4(_t, vl);
        vfloat32m4_t _y = vfrdiv_vf_f32m4(_pow, -1.0f, vl);
        _y = vfadd_vf_f32m4(_y, 1.0f, vl);
        _y = vfmul_vf_f32m4_m(_mask, _y, _y, -1.0f, vl);

        vse32_v_f32m4(output_data, _y, vl);
        output_data += vl;
        size -= vl;
    }

    output->layout = input->layout;
    output->dim_count = input->dim_count;
    for (int i = 0; i < output->dim_count; i++) {
        output->dim[i] = input->dim[i];
    }
    return CSINN_TRUE;
}
