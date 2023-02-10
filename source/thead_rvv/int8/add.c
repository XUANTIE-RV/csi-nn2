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

#include "shl_thead_rvv.h"

/*************************************************************
    note: VLEN = 128/256
*************************************************************/

/************************************************************************************
 * (1) s2(q2-z2) = s0(q0-z0) + s1(q1-z1)
 * (2) q2 = s0/s2(q0-z0) + s1/s2(q1-z1) + z2
 ***********************************************************************************/
static void elementwise_add_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                 struct csinn_tensor *output)
{
    int8_t *input0_data = (int8_t *)input0->data;
    int8_t *input1_data = (int8_t *)input1->data;
    int8_t *output_data = (int8_t *)output->data;
    int32_t zero_point0 = input0->qinfo->zero_point;
    int32_t zero_point1 = input1->qinfo->zero_point;
    int32_t zero_point2 = output->qinfo->zero_point;
    int32_t mult0 = input0->qinfo->multiplier;
    int32_t mult1 = input1->qinfo->multiplier;
    int32_t shift0 = input0->qinfo->shift;
    int32_t shift1 = input1->qinfo->shift;
    int64_t size = csinn_tensor_size(output);

    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _in0 = vle8_v_i8m1(input0_data, vl);
        vint8m1_t _in1 = vle8_v_i8m1(input1_data, vl);
        vint16m2_t _in0_w = vwadd_vx_i16m2(_in0, 0, vl);
        vint16m2_t _in1_w = vwadd_vx_i16m2(_in1, 0, vl);
        vint32m4_t _q0_z0 = vwsub_vx_i32m4(_in0_w, zero_point0, vl);
        vint32m4_t _q1_z1 = vwsub_vx_i32m4(_in1_w, zero_point1, vl);

        int32_t shift_tmp0 = 0, shift_tmp1 = 0;
        if (shift0 < 0) {
            shift_tmp0 = -shift0 - 1;
        } else {
            _q0_z0 = vsll_vx_i32m4(_q0_z0, shift0 + 2, vl);
            shift_tmp0 = 1;
        }

        if (shift1 < 0) {
            shift_tmp1 = -shift1 - 1;
        } else {
            _q1_z1 = vsll_vx_i32m4(_q1_z1, shift1 + 2, vl);
            shift_tmp1 = 1;
        }

        vint32m4_t _mulh0 = vmulh_vx_i32m4(_q0_z0, mult0, vl);
        vint32m4_t _mulh1 = vmulh_vx_i32m4(_q1_z1, mult1, vl);

        _mulh0 = vssra_vx_i32m4(_mulh0, shift_tmp0, vl);
        _mulh1 = vssra_vx_i32m4(_mulh1, shift_tmp1, vl);

        vint32m4_t _res0 = vadd_vv_i32m4(_mulh0, _mulh1, vl);
        _res0 = vadd_vx_i32m4(_res0, zero_point2, vl);
        vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);
        vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);
        vse8_v_i8m1(output_data, _res2, vl);

        input0_data += vl;
        input1_data += vl;
        output_data += vl;
        size -= vl;
    }
}

static void elementwise_add_int8_trans_fp16(struct csinn_tensor *input0,
                                            struct csinn_tensor *input1,
                                            struct csinn_tensor *output)
{
    int8_t *input0_data = (int8_t *)input0->data;
    int8_t *input1_data = (int8_t *)input1->data;
    int8_t *output_data = (int8_t *)output->data;
    int32_t zero_point0 = input0->qinfo->zero_point;
    int32_t zero_point1 = input1->qinfo->zero_point;
    int32_t zero_point2 = output->qinfo->zero_point;
    float real_s0 = input0->qinfo->scale / output->qinfo->scale;
    float real_s1 = input1->qinfo->scale / output->qinfo->scale;

    int64_t size = csinn_tensor_size(output);
    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _in0 = vle8_v_i8m1(input0_data, vl);
        vint8m1_t _in1 = vle8_v_i8m1(input1_data, vl);
        vint16m2_t _in0w = vwsub_vx_i16m2(_in0, input0->qinfo->zero_point, vl);
        vint16m2_t _in1w = vwsub_vx_i16m2(_in1, input1->qinfo->zero_point, vl);
        vfloat16m2_t _in0f = vfcvt_f_x_v_f16m2(_in0w, vl);
        vfloat16m2_t _in1f = vfcvt_f_x_v_f16m2(_in1w, vl);
        vfloat16m2_t _tmp0 = vfmul_vf_f16m2(_in0f, real_s0, vl);  // s0/s2(q0-z0)
        vfloat16m2_t _tmp1 = vfmul_vf_f16m2(_in1f, real_s1, vl);  // s1/s2(q1-z1)
        vfloat16m2_t _sumf = vfadd_vv_f16m2(_tmp0, _tmp1, vl);
        vint16m2_t _res = vfcvt_x_f_v_i16m2(_sumf, vl);
        _res = vadd_vx_i16m2(_res, zero_point2, vl);
        vse8_v_i8m1(output_data, vnclip_wx_i8m1(_res, 0, vl), vl);
        input0_data += vl;
        input1_data += vl;
        output_data += vl;
        size -= vl;
    }
}

/************************************************************************************
 * (1) q2 = s0/s2(q0-z0) + s1/s2(q1-z1) + z2
 * (2) q2 = (q0-z0) + s1/s2(q1-z1) + z2
 * (3) ps: s0/s2=1
 ***********************************************************************************/
static void broadcast_single_1_add_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                        struct csinn_tensor *output)
{
    int8_t *input0_data = (int8_t *)input0->data;
    int8_t *input1_data = (int8_t *)input1->data;
    int8_t *output_data = (int8_t *)output->data;
    float s1 = input1->qinfo->scale;
    float s2 = output->qinfo->scale;
    int32_t zero_point0 = input0->qinfo->zero_point;
    int32_t zero_point1 = input1->qinfo->zero_point;
    int32_t zero_point2 = output->qinfo->zero_point;
    int32_t q1_z1 = (int32_t)(s1 / s2 * (input1_data[0] - zero_point1));
    int32_t q1_z1_z2 = q1_z1 + zero_point2;

    int64_t size = csinn_tensor_size(output);
    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _in0 = vle8_v_i8m1(input0_data, vl);
        vint16m2_t _in0_w = vwadd_vx_i16m2(_in0, 0, vl);
        vint32m4_t _q0_z0 = vwsub_vx_i32m4(_in0_w, zero_point0, vl);
        vint32m4_t _res0 = vadd_vx_i32m4(_q0_z0, q1_z1_z2, vl);
        vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);
        vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);
        vse8_v_i8m1(output_data, _res2, vl);
        input0_data += vl;
        output_data += vl;
        size -= vl;
    }
}

int shl_rvv_add_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int64_t in_size0 = csinn_tensor_size(input0);
    int64_t in_size1 = csinn_tensor_size(input1);
    int64_t out_size = csinn_tensor_size(output);

    // TODO: move to init api
    float real_scale0 = input0->qinfo->scale / output->qinfo->scale;
    float real_scale1 = input1->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale0, &input0->qinfo->multiplier, &input0->qinfo->shift);
    shl_quantize_multiplier(real_scale1, &input1->qinfo->multiplier, &input1->qinfo->shift);

    bool is_elementwise =
        (in_size0 == out_size) && (in_size1 == out_size) && (input0->layout == input1->layout);

    if (is_elementwise) {
        output->layout = input0->layout;
        output->dim_count = input0->dim_count;
        for (int i = 0; i < output->dim_count; i++) {
            output->dim[i] = input0->dim[i];
        }
        elementwise_add_int8_trans_fp16(input0, input1, output);
    } else if (in_size1 == 1) {
        output->layout = input0->layout;
        output->dim_count = input0->dim_count;
        for (int i = 0; i < output->dim_count; i++) {
            output->dim[i] = input0->dim[i];
        }
        broadcast_single_1_add_int8(input0, input1, output);
    } else {
        /* TODO: recursive opt */
        return shl_ref_add_quant(input0, input1, output, params);
    }
    return CSINN_TRUE;
}
