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

static int tail_coincide(struct csinn_tensor *input0, struct csinn_tensor *input1)
{
    int flag = 1;
    int i = 0, j = 0;
    for (i = input1->dim_count - 1, j = input0->dim_count - 1; i >= 0; i--, j--) {
        if (input0->dim[j] != input1->dim[i]) {
            flag = 0;
            break;
        }
    }
    flag = 1;
    for (; i >= 0; i--) {
        if (input1->dim[i] != 1) {
            flag = 0;
            break;
        }
    }
    return flag;
}

/************************************************************************************
    (1) s3*(q3-z3) = s1*(q1-z1) * s2*(q2-z2)
    (2) q3 = [ (q1-z1) * (q2-z2) * (s1*s2/s3) ] + z3
    (3) output->qinfo->mulitipler means mult of s1*s2/s3 and output->qinfo->shift represents the
right shift(>0)
    TODO: broadcast mul
    note: if input1 is const, support per-channel quantization
************************************************************************************/
static void element_mul_int8(int8_t *input0, int8_t *input1, int8_t *output, int size, int32_t mult,
                             int32_t shift, int32_t zero_point0, int32_t zero_point1,
                             int32_t zero_point2)
{
    int32_t z1z2 = zero_point0 * zero_point1;
    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _in0 = vle8_v_i8m1(input0, vl);
        vint8m1_t _in1 = vle8_v_i8m1(input1, vl);

        vint16m2_t _q1q2 = vwmul_vv_i16m2(_in0, _in1, vl);
        vint16m2_t _q1z2 = vwmul_vx_i16m2(_in0, (int8_t)zero_point1, vl);
        vint16m2_t _q2z1 = vwmul_vx_i16m2(_in1, (int8_t)zero_point0, vl);

        vint32m4_t _res = vwsub_vv_i32m4(_q1q2, _q1z2, vl);  // q1q2 - q1z2
        _res = vwsub_wv_i32m4(_res, _q2z1, vl);              // q1q2 - q1z2 - q2z1
        _res = vadd_vx_i32m4(_res, z1z2, vl);                // q1q2 - q1z2 - q2z1 + z1z2
        input0 += vl;
        input1 += vl;
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
        vse8_v_i8m1(output, _res2, vl);
        output += vl;
        size -= vl;
    }
}

int shl_rvv_mul_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    // return shl_ref_mul_quant(input0, input1, output, params);
    int8_t *input0_data = (int8_t *)input0->data;
    int8_t *input1_data = (int8_t *)input1->data;
    int8_t *output_data = (int8_t *)output->data;

    int in_size0 = csinn_tensor_size(input0);
    int in_size1 = csinn_tensor_size(input1);
    int out_size = csinn_tensor_size(output);

    // TODO: move to init api
    for (int q = 0; q < input1->quant_channel; q++) {
        float real_scale = input0->qinfo->scale * input1->qinfo[q].scale / output->qinfo->scale;
        shl_quantize_multiplier(real_scale, &input1->qinfo[q].multiplier, &input1->qinfo[q].shift);
    }

    if (in_size1 == 1) {
        // q3 = [ (q1-z1) * (q2-z2) * (s1*s2/s3) ] + z3 = (q1-z1) + z3
        int size = out_size;
        int32_t zero_point0 = input0->qinfo->zero_point;
        int32_t zero_point2 = output->qinfo->zero_point;
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
    } else if (in_size0 == in_size1) {
        int outer_size = input1->quant_channel;
        int inner_size = in_size1 / outer_size;
        for (int c = 0; c < outer_size; c++) {
            element_mul_int8(input0_data, input1_data, output_data, inner_size,
                             input1->qinfo[c].multiplier, input1->qinfo[c].shift,
                             input0->qinfo->zero_point, input1->qinfo[c].zero_point,
                             output->qinfo->zero_point);
        }
    } else if (tail_coincide(input0, input1)) {
        int inner_size = in_size1;
        int outer_size = out_size / in_size1;
        for (int i = 0; i < outer_size; i++) {
            element_mul_int8(input0_data, input1_data, output_data, inner_size,
                             input1->qinfo[0].multiplier, input1->qinfo[0].shift,
                             input0->qinfo->zero_point, input1->qinfo->zero_point,
                             output->qinfo->zero_point);
            input0_data += inner_size;
            output_data += inner_size;
        }
    } else {
        int8_t *in0_data_b = shl_mem_alloc(out_size * sizeof(int8_t));
        int8_t *in1_data_b = shl_mem_alloc(out_size * sizeof(int8_t));

        struct csinn_tensor *b_input0 = csinn_alloc_tensor(NULL);
        struct csinn_tensor *b_input1 = csinn_alloc_tensor(NULL);
        csinn_tensor_copy(b_input0, output);
        csinn_tensor_copy(b_input1, output);
        b_input0->data = in0_data_b;
        b_input1->data = in1_data_b;

        shl_ref_broadcast_to_shape_quant(input0, b_input0, output->dim, output->dim_count);
        shl_ref_broadcast_to_shape_quant(input1, b_input1, output->dim, output->dim_count);

        input0_data = b_input0->data;
        input1_data = b_input1->data;

        element_mul_int8(input0_data, input1_data, output_data, out_size,
                         input1->qinfo[0].multiplier, input1->qinfo[0].shift,
                         input0->qinfo->zero_point, input1->qinfo->zero_point,
                         output->qinfo->zero_point);

        shl_mem_free(in0_data_b);
        shl_mem_free(in1_data_b);
        csinn_free_tensor(b_input0);
        csinn_free_tensor(b_input1);
    }
    return CSINN_TRUE;
}
