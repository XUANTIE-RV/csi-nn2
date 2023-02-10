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

/************************************************************************************
 * (1) s2*(q2-z2) = s0*(q0-z0) * s1*(q1-z1)
 * (2) q2 = [ (q0-z0) * (q1-z1) * (s0*s1/s2) ] + z2
 * (3) output->qinfo->mulitipler(shift) means mult(shift) of s0*s1/s2
 ***********************************************************************************/
static void elementwise_mul_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                 struct csinn_tensor *output)
{
    int8_t *input0_data = (int8_t *)input0->data;
    int8_t *input1_data = (int8_t *)input1->data;
    int8_t *output_data = (int8_t *)output->data;
    int32_t mult = output->qinfo->multiplier;
    int32_t shift = output->qinfo->shift;
    int32_t zero_point0 = input0->qinfo->zero_point;
    int32_t zero_point1 = input1->qinfo->zero_point;
    int32_t zero_point2 = output->qinfo->zero_point;
    int32_t z1z2 = zero_point0 * zero_point1;

    int64_t size = csinn_tensor_size(output);
    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _in0 = vle8_v_i8m1(input0_data, vl);
        vint8m1_t _in1 = vle8_v_i8m1(input1_data, vl);

        vint16m2_t _q1q2 = vwmul_vv_i16m2(_in0, _in1, vl);
        vint16m2_t _q1z2 = vwmul_vx_i16m2(_in0, (int8_t)zero_point1, vl);
        vint16m2_t _q2z1 = vwmul_vx_i16m2(_in1, (int8_t)zero_point0, vl);

        vint32m4_t _res = vwsub_vv_i32m4(_q1q2, _q1z2, vl);  // q1q2 - q1z2
        _res = vwsub_wv_i32m4(_res, _q2z1, vl);              // q1q2 - q1z2 - q2z1
        _res = vadd_vx_i32m4(_res, z1z2, vl);                // q1q2 - q1z2 - q2z1 + z1z2
        input0_data += vl;
        input1_data += vl;
        // FIXME: precision error
        vint32m4_t _mulh = vmulh_vx_i32m4(_res, mult, vl);
        if (shift < 0) {
            _res = vssra_vx_i32m4(_mulh, -shift - 1, vl);
        } else {
            _res = vsll_vx_i32m4(_mulh, shift + 1, vl);
        }

        _res = vadd_vx_i32m4(_res, zero_point2, vl);
        vint16m2_t _res1 = vnclip_wx_i16m2(_res, 0, vl);
        vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);
        vse8_v_i8m1(output_data, _res2, vl);
        output_data += vl;
        size -= vl;
    }
}

static void elementwise_mul_int8_trans_fp16(struct csinn_tensor *input0,
                                            struct csinn_tensor *input1,
                                            struct csinn_tensor *output)
{
    int8_t *input0_data = (int8_t *)input0->data;
    int8_t *input1_data = (int8_t *)input1->data;
    int8_t *output_data = (int8_t *)output->data;
    int32_t zero_point0 = input0->qinfo->zero_point;
    int32_t zero_point1 = input1->qinfo->zero_point;
    int32_t zero_point2 = output->qinfo->zero_point;
    float real_scale = input0->qinfo->scale * input1->qinfo->scale / output->qinfo->scale;

    int64_t size = csinn_tensor_size(output);
    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _in0 = vle8_v_i8m1(input0_data, vl);
        vint8m1_t _in1 = vle8_v_i8m1(input1_data, vl);
        vint16m2_t _in0w = vwsub_vx_i16m2(_in0, zero_point0, vl);
        vint16m2_t _in1w = vwsub_vx_i16m2(_in1, zero_point1, vl);
        vfloat16m2_t _in0f = vfcvt_f_x_v_f16m2(_in0w, vl);
        vfloat16m2_t _in1f = vfcvt_f_x_v_f16m2(_in1w, vl);
        vfloat16m2_t _mulf = vfmul_vv_f16m2(_in0f, _in1f, vl);
        _mulf = vfmul_vf_f16m2(_mulf, real_scale, vl);
        vint16m2_t _res = vfcvt_x_f_v_i16m2(_mulf, vl);
        _res = vadd_vx_i16m2(_res, zero_point2, vl);
        vse8_v_i8m1(output_data, vnclip_wx_i8m1(_res, 0, vl), vl);
        input0_data += vl;
        input1_data += vl;
        output_data += vl;
        size -= vl;
    }
}

/************************************************************************************
 * (1) q2 = [ (q0-z0) * (q1-z1) * (s0*s1/s2) ] + z2
 * (2) q2 = (q0-z0) + z2
 * (3) ps: (q1-z1) * (s0*s1/s2) = 1
 ***********************************************************************************/
static void broadcast_single_1_mul_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                        struct csinn_tensor *output)
{
    int8_t *input0_data = (int8_t *)input0->data;
    int8_t *input1_data = (int8_t *)input1->data;
    int8_t *output_data = (int8_t *)output->data;
    int32_t zero_point0 = input0->qinfo->zero_point;
    int32_t zero_point2 = output->qinfo->zero_point;

    int64_t size = csinn_tensor_size(output);
    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _in0 = vle8_v_i8m1(input0_data, vl);
        vint16m2_t _q1_z1 = vwsub_vx_i16m2(_in0, zero_point0, vl);
        vint16m2_t _res0 = vadd_vx_i16m2(_q1_z1, zero_point2, vl);
        vint8m1_t _res1 = vnclip_wx_i8m1(_res0, 0, vl);
        vse8_v_i8m1(output_data, _res1, vl);
        input0_data += vl;
        output_data += vl;
        size -= vl;
    }
}

int shl_rvv_mul_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int64_t in_size0 = csinn_tensor_size(input0);
    int64_t in_size1 = csinn_tensor_size(input1);
    int64_t out_size = csinn_tensor_size(output);

    // TODO: move to init api
    float real_scale = input0->qinfo->scale * input1->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale, &output->qinfo->multiplier, &output->qinfo->shift);

    bool is_elementwise =
        (in_size0 == out_size) && (in_size1 == out_size) && (input0->layout == input1->layout);

    if (is_elementwise) {
        output->layout = input0->layout;
        output->dim_count = input0->dim_count;
        for (int i = 0; i < output->dim_count; i++) {
            output->dim[i] = input0->dim[i];
        }
        elementwise_mul_int8_trans_fp16(input0, input1, output);
    } else if (in_size1 == 1) {
        output->layout = input0->layout;
        output->dim_count = input0->dim_count;
        for (int i = 0; i < output->dim_count; i++) {
            output->dim[i] = input0->dim[i];
        }
        broadcast_single_1_mul_int8(input0, input1, output);
    } else {
        /* TODO: recursive opt */
        return shl_ref_mul_quant(input0, input1, output, params);
    }
    return CSINN_TRUE;
}
