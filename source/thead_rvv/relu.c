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

/*************************************************************
    note: VLEN = 128/256 ...
*************************************************************/
int csi_nn_rvv_relu_fp32(struct csi_tensor *input, struct csi_tensor *output,
                         struct relu_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    int vl = vsetvl_e32m2(size);  // vl=8 if vlen=128

    int i = 0;
    for (; i + vl - 1 < size; i += vl) {
        vfloat32m2_t _input = vle32_v_f32m2(input_data, vl);
        input_data += vl;
        vfloat32m2_t _output = vfmax_vf_f32m2(_input, 0.0f, vl);
        vse32_v_f32m2(output_data, _output, vl);
        output_data += vl;
    }
    if (i < size) {
        vl = vsetvl_e32m2(size & (vl - 1));  // ???
        vfloat32m2_t _input = vle32_v_f32m2(input_data, vl);
        vfloat32m2_t _output = vfmax_vf_f32m2(_input, 0.0f, vl);
        vse32_v_f32m2(output_data, _output, vl);
    }
    return CSINN_TRUE;
}

int csi_nn_rvv_relu_fp16(struct csi_tensor *input, struct csi_tensor *output,
                         struct relu_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    int vl = vsetvl_e16m2(size);

    int i = 0;
    for (; i + vl - 1 < size; i += vl) {
        vfloat16m2_t _input = vle16_v_f16m2(input_data, vl);
        input_data += vl;
        vfloat16m2_t _output = vfmax_vf_f16m2(_input, 0.0f, vl);
        vse16_v_f16m2(output_data, _output, vl);
        output_data += vl;
    }
    if (i < size) {
        vl = vsetvl_e16m2(size & (vl - 1));
        vfloat16m2_t _input = vle16_v_f16m2(input_data, vl);
        vfloat16m2_t _output = vfmax_vf_f16m2(_input, 0.0f, vl);
        vse16_v_f16m2(output_data, _output, vl);
    }
    return CSINN_TRUE;
}

/************************************************************************************
 * s2(q2 - z2) = relu{ s1(q1 - z1) }
 * q2 = (q1 - z1) * s1/s2 + z2
 *
 * note：relu 一般接在全连接/卷积后面，可以直接和全连接/卷积 融合
 ************************************************************************************/
int csi_nn_rvv_relu_int8(struct csi_tensor *input, struct csi_tensor *output,
                         struct relu_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    // TODO: move to init api
    // real_scale > 1 =>  output->qinfo->shift > 0  ==> shift left
    float real_scale = input->qinfo->scale / output->qinfo->scale;
    csi_quantize_multiplier(real_scale, &output->qinfo->multiplier, &output->qinfo->shift);

    int size = csi_tensor_size(input);
    while (size > 0) {
        int vl = vsetvl_e8m1(size);

        vint8m1_t _input = vle8_v_i8m1(input_data, vl);
        vint16m2_t _input1 = vwadd_vx_i16m2(_input, 0, vl);   // widden 8->16
        vint32m4_t _input2 = vwadd_vx_i32m4(_input1, 0, vl);  // widden 16->32

        vint32m4_t _tmp = vsub_vx_i32m4(_input2, input->qinfo->zero_point, vl);
        // mulh 无 round 过程, 左移时多移1位，mulh 后再用带round的右移1位来实现类似round的功能
        _tmp = vsll_vx_i32m4(_tmp, output->qinfo->shift + 2, vl);
        vint32m4_t _mulh = vmulh_vx_i32m4(_tmp, output->qinfo->multiplier, vl);
        _mulh = vssra_vx_i32m4(_mulh, 1, vl);

        vint32m4_t _res0 = vadd_vx_i32m4(_mulh, output->qinfo->zero_point, vl);  // +z2 (z2 = -128)
        vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);                        // narrow 32->16
        vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);                          // narrow 16->8

        vse8_v_i8m1(output_data, _res2, vl);
        input_data += vl;
        output_data += vl;
        size -= vl;
    }
    return CSINN_TRUE;
}
