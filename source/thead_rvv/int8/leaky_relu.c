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

/*************************************************************
    note: VLEN = 128/256 ...
*************************************************************/

/*********************************************************************
 * s2 * (q2 - z2) = leaky_relu{ s1 * (q1 - z1) }
 * if (q1 >= z1)  q2 = s1/s2 * (q1 - z1) + z2
 * else q2 = s1/s2 * alpha * (q1 -z1) + z2
 * ******************************************************************/
int shl_rvv_leaky_relu_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_relu_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    // TODO: move to init api
    float real_scale0 = input->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale0, &output->qinfo->multiplier, &output->qinfo->shift);

    int size = csinn_tensor_size(input);
    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _input = vle8_v_i8m1(input_data, vl);
        vint16m2_t _input1 = vwadd_vx_i16m2(_input, 0, vl);   // widden 8->16
        vint32m4_t _input2 = vwadd_vx_i32m4(_input1, 0, vl);  // widden 16->32

        vint32m4_t _tmp = vsub_vx_i32m4(_input2, input->qinfo->zero_point, vl);

        _tmp = vsll_vx_i32m4(_tmp, output->qinfo->shift + 2, vl);
        vint32m4_t _mulh = vmulh_vx_i32m4(_tmp, output->qinfo->multiplier, vl);
        _mulh = vssra_vx_i32m4(_mulh, 1, vl);

        vbool8_t _mask = vmslt_vx_i32m4_b8(_input2, input->qinfo->zero_point, vl);
        vint32m4_t _mulh_neg = vmulh_vx_i32m4_m(_mask, _mulh, _mulh, params->n_multiplier, vl);
        if (params->n_shift < 0) {
            _mulh_neg = vssra_vx_i32m4_m(_mask, _mulh, _mulh_neg, -params->n_shift - 1, vl);
        } else {
            _mulh_neg = vsll_vx_i32m4_m(_mask, _mulh, _mulh_neg, params->n_shift + 1, vl);
        }

        vint32m4_t _res0 = vadd_vx_i32m4(_mulh_neg, output->qinfo->zero_point, vl);
        vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);
        vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);

        vse8_v_i8m1(output_data, _res2, vl);
        input_data += vl;
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
