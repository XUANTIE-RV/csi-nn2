/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

static inline void div_vv_i8_trans_f16(int8_t *in0, int8_t *in1, int8_t *out, int32_t size,
                                       float *scale, int32_t *zero_point)
{
    int32_t z0 = zero_point[0];
    int32_t z1 = zero_point[1];
    int32_t z2 = zero_point[2];
    float real_scale = scale[0] / scale[1] / scale[2];

    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _a = vle8_v_i8m1(in0, vl);
        vint8m1_t _b = vle8_v_i8m1(in1, vl);
        vint16m2_t _a_w = vwsub_vx_i16m2(_a, z0, vl);
        vint16m2_t _b_w = vwsub_vx_i16m2(_b, z1, vl);
        vfloat16m2_t _a_f = vfcvt_f_x_v_f16m2(_a_w, vl);
        vfloat16m2_t _b_f = vfcvt_f_x_v_f16m2(_b_w, vl);
        vfloat16m2_t _divf = vfdiv_vv_f16m2(_a_f, _b_f, vl);
        _divf = vfmul_vf_f16m2(_divf, real_scale, vl);
        vint16m2_t _res = vfcvt_x_f_v_i16m2(_divf, vl);
        _res = vadd_vx_i16m2(_res, z2, vl);
        vse8_v_i8m1(out, vnclip_wx_i8m1(_res, 0, vl), vl);
        in0 += vl;
        in1 += vl;
        out += vl;
        size -= vl;
    }
}

static inline void div_vx_i8_trans_f16(int8_t *in0, int8_t *in1, int8_t *out, int32_t size,
                                       float *scale, int32_t *zero_point)
{
    int32_t z0 = zero_point[0];
    int32_t z1 = zero_point[1];
    int32_t z2 = zero_point[2];
    float real_scale = scale[0] / scale[1] / scale[2];
    float b_f = in1[0] - z1;

    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _a = vle8_v_i8m1(in0, vl);
        vint16m2_t _a_w = vwsub_vx_i16m2(_a, z0, vl);
        vfloat16m2_t _a_f = vfcvt_f_x_v_f16m2(_a_w, vl);
        vfloat16m2_t _divf = vfdiv_vf_f16m2(_a_f, b_f, vl);
        _divf = vfmul_vf_f16m2(_divf, real_scale, vl);
        vint16m2_t _res = vfcvt_x_f_v_i16m2(_divf, vl);
        _res = vadd_vx_i16m2(_res, z2, vl);
        vse8_v_i8m1(out, vnclip_wx_i8m1(_res, 0, vl), vl);
        in0 += vl;
        out += vl;
        size -= vl;
    }
}

static inline void div_xv_i8_trans_f16(int8_t *in0, int8_t *in1, int8_t *out, int32_t size,
                                       float *scale, int32_t *zero_point)
{
    int32_t z0 = zero_point[0];
    int32_t z1 = zero_point[1];
    int32_t z2 = zero_point[2];
    float real_scale = scale[0] / scale[1] / scale[2];
    float a_f = in0[0] - z0;

    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _b = vle8_v_i8m1(in1, vl);
        vint16m2_t _b_w = vwsub_vx_i16m2(_b, z1, vl);
        vfloat16m2_t _b_f = vfcvt_f_x_v_f16m2(_b_w, vl);
        vfloat16m2_t _divf = vfrdiv_vf_f16m2(_b_f, a_f, vl);
        _divf = vfmul_vf_f16m2(_divf, real_scale, vl);
        vint16m2_t _res = vfcvt_x_f_v_i16m2(_divf, vl);
        _res = vadd_vx_i16m2(_res, z2, vl);
        vse8_v_i8m1(out, vnclip_wx_i8m1(_res, 0, vl), vl);
        in1 += vl;
        out += vl;
        size -= vl;
    }
}

void *div_cb_int8[] = {
    [CSINN_BROADCAST_VV] = div_vv_i8_trans_f16,
    [CSINN_BROADCAST_VS] = div_vx_i8_trans_f16,
    [CSINN_BROADCAST_SV] = div_xv_i8_trans_f16,
};

int shl_rvv_div_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    return shl_rvv_binary_op_broadcast_int8(input0, input1, output, div_cb_int8);
}
