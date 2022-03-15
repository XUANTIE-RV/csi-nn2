/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* CSI-NN2 version 1.12.x */

#include "csi_thead_rvv.h"

int csi_nn_rvv_mul_fp32(struct csi_tensor *input0, struct csi_tensor *input1,
                        struct csi_tensor *output, struct diso_params *params)
{
    return CSINN_TRUE;
}

int csi_nn_rvv_mul_fp16(struct csi_tensor *input0, struct csi_tensor *input1,
                        struct csi_tensor *output, struct diso_params *params)
{
    return CSINN_TRUE;
}

/************************************************************************************
    (1) s3*(q3-z3) = s1*(q1-z1) * s2*(q2-z2)
    (2) q3 = [ (q1-z1) * (q2-z2) * (s1*s2/s3) ]  +  z3
    (3) output->qinfo->mulitipler means mult of s1*s2/s3 and output->qinfo->shift represents the
right shift(>0)
    TODO: broadcast mul
    note: if input1 is const, support per-channel quantization
************************************************************************************/
int csi_nn_rvv_mul_int8(struct csi_tensor *input0, struct csi_tensor *input1,
                        struct csi_tensor *output, struct diso_params *params)
{
    int8_t *input0_data = (int8_t *)input0->data;
    int8_t *input1_data = (int8_t *)input1->data;
    int8_t *output_data = (int8_t *)output->data;

    int in_size0 = csi_tensor_size(input0);
    int in_size1 = csi_tensor_size(input1);
    int out_size = csi_tensor_size(output);

    // TODO: move to init api
    for (int q = 0; q < input1->quant_channel; q++) {
        float real_scale = input0->qinfo->scale * input1->qinfo[q].scale / output->qinfo->scale;
        csi_quantize_multiplier(real_scale, &input1->qinfo[q].multiplier, &input1->qinfo[q].shift);
    }

    if (in_size0 == in_size1) {
        int i = 0;
        int packn = csrr_vlenb() / sizeof(int8_t);
        int outer_size = input1->quant_channel;
        int inner_size = in_size1 / outer_size;
        for (int c = 0; c < outer_size; c++) {
            int32_t z1z2 = input0->qinfo->zero_point * input1->qinfo[c].zero_point;
            int size = inner_size;
            while (size > 0) {
                int vl = vsetvl_e8m1(size);
                vint8m1_t _in0 = vle8_v_i8m1(input0_data, vl);
                vint8m1_t _in1 = vle8_v_i8m1(input1_data, vl);

                vint16m2_t _q1q2 = vwmul_vv_i16m2(_in0, _in1, vl);
                vint16m2_t _q1z2 = vwmul_vx_i16m2(_in0, (int8_t)input1->qinfo[c].zero_point, vl);
                vint16m2_t _q2z1 = vwmul_vx_i16m2(_in1, (int8_t)input0->qinfo->zero_point, vl);

                vint32m4_t _res = vwsub_vv_i32m4(_q1q2, _q1z2, vl);  // q1q2 - q1z2
                _res = vwsub_wv_i32m4(_res, _q2z1, vl);              // q1q2 - q1z2 - q2z1
                _res = vadd_vx_i32m4(_res, z1z2, vl);                // q1q2 - q1z2 - q2z1 + z1z2
                input0_data += vl;
                input1_data += vl;
                // FIXME: precision error
                vint32m4_t _mulh = vmulh_vx_i32m4(_res, input1->qinfo[c].multiplier, vl);
                if (input1->qinfo[c].shift < 0) {
                    _res = vssra_vx_i32m4(_mulh, -input1->qinfo[c].shift - 1, vl);
                } else {
                    _res = vsll_vx_i32m4(_mulh, input1->qinfo[c].shift + 1, vl);
                }

                _res = vadd_vx_i32m4(_res, output->qinfo->zero_point, vl);
                vint16m2_t _res1 = vnclip_wx_i16m2(_res, 0, vl);
                vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);
                vse8_v_i8m1(output_data, _res2, vl);
                output_data += vl;
                size -= vl;
            }
        }
    } else {
        csi_debug_error("Only support elementwise mul on RVV CPU\n");
    }
    return CSINN_TRUE;
}
