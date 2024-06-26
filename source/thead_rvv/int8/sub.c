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

static inline void sub_vv_i8_trans_f16(int8_t *in0, int8_t *in1, int8_t *out, int32_t size,
                                       float *scale, int32_t *zero_point)
{
    float s0_s2 = scale[0] / scale[2];
    float s1_s2 = scale[1] / scale[2];
    int32_t z0 = zero_point[0];
    int32_t z1 = zero_point[1];
    int32_t z2 = zero_point[2];

    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _a = vle8_v_i8m1(in0, vl);
        vint8m1_t _b = vle8_v_i8m1(in1, vl);
        vint16m2_t _a_w = vwsub_vx_i16m2(_a, z0, vl);
        vint16m2_t _b_w = vwsub_vx_i16m2(_b, z1, vl);
        vfloat16m2_t _a_f = vfcvt_f_x_v_f16m2(_a_w, vl);
        vfloat16m2_t _b_f = vfcvt_f_x_v_f16m2(_b_w, vl);
        vfloat16m2_t _tmp0 = vfmul_vf_f16m2(_a_f, s0_s2, vl);  // s0/s2(q0-z0)
        vfloat16m2_t _tmp1 = vfmul_vf_f16m2(_b_f, s1_s2, vl);  // s1/s2(q1-z1)
        vfloat16m2_t _subf = vfsub_vv_f16m2(_tmp0, _tmp1, vl);
        vint16m2_t _res = vfcvt_x_f_v_i16m2(_subf, vl);
        _res = vadd_vx_i16m2(_res, z2, vl);
        vse8_v_i8m1(out, vnclip_wx_i8m1(_res, 0, vl), vl);
        in0 += vl;
        in1 += vl;
        out += vl;
        size -= vl;
    }
}

static inline void sub_vx_i8_trans_f16(int8_t *in0, int8_t *in1, int8_t *out, int32_t size,
                                       float *scale, int32_t *zero_point)
{
    float s0_s2 = scale[0] / scale[2];
    int32_t z0 = zero_point[0];
    int32_t z1 = zero_point[1];
    int32_t z2 = zero_point[2];
    float q1_z1 = scale[1] / scale[2] * (in1[0] - z1);  // s1/s2(q1-z1)
    float q1_z1_z2 = q1_z1 - z2;

    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _a = vle8_v_i8m1(in0, vl);
        vint16m2_t _a_w = vwsub_vx_i16m2(_a, z0, vl);
        vfloat16m2_t _a_f = vfcvt_f_x_v_f16m2(_a_w, vl);
        vfloat16m2_t _tmp0 = vfmul_vf_f16m2(_a_f, s0_s2, vl);  // s0/s2(q0-z0)
        vfloat16m2_t _subf = vfsub_vf_f16m2(_tmp0, q1_z1_z2, vl);
        vint16m2_t _res = vfcvt_x_f_v_i16m2(_subf, vl);
        vse8_v_i8m1(out, vnclip_wx_i8m1(_res, 0, vl), vl);
        in0 += vl;
        out += vl;
        size -= vl;
    }
}

static inline void sub_xv_i8_trans_f16(int8_t *in0, int8_t *in1, int8_t *out, int32_t size,
                                       float *scale, int32_t *zero_point)
{
    float s1_s2 = scale[1] / scale[2];
    int32_t z0 = zero_point[0];
    int32_t z1 = zero_point[1];
    int32_t z2 = zero_point[2];
    float q0_z0 = scale[0] / scale[2] * (in0[0] - z0);  // s0/s2(q0-z0)
    float q0_z0_z2 = q0_z0 + z2;

    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _b = vle8_v_i8m1(in1, vl);
        vint16m2_t _b_w = vwsub_vx_i16m2(_b, z1, vl);
        vfloat16m2_t _b_f = vfcvt_f_x_v_f16m2(_b_w, vl);
        vfloat16m2_t _tmp1 = vfmul_vf_f16m2(_b_f, s1_s2, vl);  // s1/s2(q1-z1)
        vfloat16m2_t _subf = vfrsub_vf_f16m2(_tmp1, q0_z0_z2, vl);
        vint16m2_t _res = vfcvt_x_f_v_i16m2(_subf, vl);
        vse8_v_i8m1(out, vnclip_wx_i8m1(_res, 0, vl), vl);
        in1 += vl;
        out += vl;
        size -= vl;
    }
}

void *sub_cb_int8[] = {
    [CSINN_BROADCAST_VV] = sub_vv_i8_trans_f16,
    [CSINN_BROADCAST_VS] = sub_vx_i8_trans_f16,
    [CSINN_BROADCAST_SV] = sub_xv_i8_trans_f16,
};

int shl_rvv_sub_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    return shl_rvv_binary_op_broadcast_int8(input0, input1, output, sub_cb_int8);
}
