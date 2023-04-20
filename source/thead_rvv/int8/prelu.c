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

/*********************************************************************
 * s3 * (q3 - z3) = prelu{ s1 * (q1 - z1), s2 * (q2 - z2) }
 * if (q1 >= z1)  q3 = s1/s3 * (q1 - z1) + z3
 * else q3 = s1*s2/s3 * (q1 - z1) * (q2 - z2) + z3
 * ******************************************************************/
int shl_rvv_prelu_int8(struct csinn_tensor *input, struct csinn_tensor *alpha,
                       struct csinn_tensor *output, struct csinn_prelu_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *alpha_data = (int8_t *)alpha->data;
    int8_t *output_data = (int8_t *)output->data;

    float real_scale0 = input->qinfo->scale / output->qinfo->scale;                        // >= 0
    float real_scale1 = input->qinfo->scale * alpha->qinfo->scale / output->qinfo->scale;  // < 0
    int32_t multiplier0, shift0, multiplier1, shift1;
    shl_quantize_multiplier(real_scale0, &multiplier0, &shift0);
    shl_quantize_multiplier(real_scale1, &multiplier1, &shift1);

    for (int n = 0; n < input->dim[0]; ++n) {
        for (int c = 0; c < input->dim[1]; ++c) {
            int32_t a = alpha_data[c] - alpha->qinfo->zero_point;
            int inner_size = input->dim[2] * input->dim[3];

            while (inner_size > 0) {
                int vl = vsetvl_e8m1(inner_size);
                vint8m1_t _input = vle8_v_i8m1(input_data, vl);
                vint16m2_t _input1 = vwadd_vx_i16m2(_input, 0, vl);
                vint32m4_t _input2 = vwadd_vx_i32m4(_input1, 0, vl);

                vbool8_t _mask0 = vmsge_vx_i32m4_b8(_input2, input->qinfo->zero_point, vl);  // >= 0
                vbool8_t _mask1 = vmslt_vx_i32m4_b8(_input2, input->qinfo->zero_point, vl);  // < 0

                vint32m4_t _tmp = vsub_vx_i32m4(_input2, input->qinfo->zero_point, vl);
                _tmp = vmul_vx_i32m4_m(_mask1, _tmp, _tmp, a, vl);
                _tmp = vsll_vx_i32m4_m(_mask0, _tmp, _tmp, shift0 + 2, vl);

                vint32m4_t _mulh = vmulh_vx_i32m4_m(_mask0, _tmp, _tmp, multiplier0, vl);
                _mulh = vmulh_vx_i32m4_m(_mask1, _mulh, _mulh, multiplier1, vl);
                _mulh = vssra_vx_i32m4_m(_mask1, _mulh, _mulh, -shift1 - 2, vl);
                _mulh = vssra_vx_i32m4(_mulh, 1, vl);

                vint32m4_t _res0 = vadd_vx_i32m4(_mulh, output->qinfo->zero_point, vl);
                vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);
                vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);

                vse8_v_i8m1(output_data, _res2, vl);
                input_data += vl;
                output_data += vl;
                inner_size -= vl;
            }
        }
    }

    return CSINN_TRUE;
}
