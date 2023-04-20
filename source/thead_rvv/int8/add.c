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
    note: VLEN = 128/256
*************************************************************/
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

// s2(q2-z2) = s0(q0-z0) + s1(q1-z1)
// q2 = s0/s2(q0-z0) + s1/s2(q1-z1) + z2
static void element_add_int8(int8_t *input0, int8_t *input1, int8_t *output, int size,
                             int32_t mult0, int32_t shift0, int32_t mult1, int32_t shift1,
                             int32_t zero_point0, int32_t zero_point1, int32_t zero_point2)
{
    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _in0 = vle8_v_i8m1(input0, vl);
        vint8m1_t _in1 = vle8_v_i8m1(input1, vl);
        vint16m2_t _in0_w = vwadd_vx_i16m2(_in0, 0, vl);
        vint16m2_t _in1_w = vwadd_vx_i16m2(_in1, 0, vl);  // widden 8 -> 16
        // vint32m4_t _in0_ww = vwadd_vx_i32m4(_in0_w, 0, vl);
        // vint32m4_t _in1_ww = vwadd_vx_i32m4(_in1_w, 0, vl);  // widden 16 -> 32

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
        vse8_v_i8m1(output, _res2, vl);

        input0 += vl;
        input1 += vl;
        output += vl;
        size -= vl;
    }
}

int shl_rvv_add_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    // return shl_ref_add_quant(input0, input1, output, params);
    int8_t *input0_data = (int8_t *)input0->data;
    int8_t *input1_data = (int8_t *)input1->data;
    int8_t *output_data = (int8_t *)output->data;

    int in_size0 = csinn_tensor_size(input0);
    int in_size1 = csinn_tensor_size(input1);
    int out_size = csinn_tensor_size(output);

    // TODO: move to init api
    float real_scale0 = input0->qinfo->scale / output->qinfo->scale;
    float real_scale1 = input1->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale0, &input0->qinfo->multiplier, &input0->qinfo->shift);
    shl_quantize_multiplier(real_scale1, &input1->qinfo->multiplier, &input1->qinfo->shift);

    if (in_size1 == 1) {
        // q2 = s0/s2(q0-z0) + s1/s2(q1-z1) + z2 = (q0-z0) + s1/s2(q1-z1) + z2
        int size = out_size;
        int32_t zero_point0 = input0->qinfo->zero_point;
        int32_t zero_point1 = input1->qinfo->zero_point;
        int32_t q1_z1 =
            (int32_t)(input1->qinfo->scale / output->qinfo->scale * (input1_data[0] - zero_point1));
        int32_t q1_z1_z2 = q1_z1 + output->qinfo->zero_point;
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
    } else if (in_size0 == in_size1) {
        element_add_int8(input0_data, input1_data, output_data, in_size0, input0->qinfo->multiplier,
                         input0->qinfo->shift, input1->qinfo->multiplier, input1->qinfo->shift,
                         input0->qinfo->zero_point, input1->qinfo->zero_point,
                         output->qinfo->zero_point);
    } else if (tail_coincide(input0, input1)) {
        int inner_size = in_size1;
        int outer_size = out_size / in_size1;
        for (int i = 0; i < outer_size; i++) {
            element_add_int8(
                input0_data, input1_data, output_data, inner_size, input0->qinfo->multiplier,
                input0->qinfo->shift, input1->qinfo->multiplier, input1->qinfo->shift,
                input0->qinfo->zero_point, input1->qinfo->zero_point, output->qinfo->zero_point);
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

        element_add_int8(input0_data, input1_data, output_data, out_size, input0->qinfo->multiplier,
                         input0->qinfo->shift, input1->qinfo->multiplier, input1->qinfo->shift,
                         input0->qinfo->zero_point, input1->qinfo->zero_point,
                         output->qinfo->zero_point);

        shl_mem_free(in0_data_b);
        shl_mem_free(in1_data_b);
        csinn_free_tensor(b_input0);
        csinn_free_tensor(b_input1);
    }
    return CSINN_TRUE;
}
