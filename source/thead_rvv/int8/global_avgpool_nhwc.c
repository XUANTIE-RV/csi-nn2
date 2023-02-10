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

int shl_rvv_global_avgpool2d_nhwc_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_pool_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    int batch = input->dim[0];
    int in_h = input->dim[1];
    int in_w = input->dim[2];
    int in_c = input->dim[3];
    int in_hw = in_h * in_w;
    __fp16 ratio = 1.0f / in_hw;
    int vl;
    int c;
    __fp16 real_scale = input->qinfo->scale / output->qinfo->scale;
    __fp16 global_z1 = (__fp16)input->qinfo->zero_point * in_hw;
    int z2 = output->qinfo->zero_point;
    ratio *= real_scale;

    __fp16 *acc_buf = (__fp16 *)shl_mem_alloc(in_c * sizeof(__fp16));

    for (int b = 0; b < batch; b++) {
        c = 0;
        while (c < in_c) {
            vl = vsetvl_e16m2(in_c - c);
            vfloat16m2_t _acc = vfmv_v_f_f16m2(-global_z1, vl);
            vse16_v_f16m2(acc_buf + c, _acc, vl);
            c += vl;
        }

        const int8_t *src = (int8_t *)input_data + b * in_h * in_w * in_c;
        for (int h = 0; h < in_h; h++) {
            for (int w = 0; w < in_w; w++) {
                const int8_t *in_ptr = src + (h * in_w + w) * in_c;
                c = 0;
                while (c < in_c) {
                    vl = vsetvl_e16m2(in_c - c);
                    vfloat16m2_t _acc = vle16_v_f16m2(acc_buf + c, vl);
                    vint8m1_t _tmp0 = vle8_v_i8m1(in_ptr + c, vl);
                    vint16m2_t _tmp1 = vwadd_vx_i16m2(_tmp0, 0, vl);
                    vfloat16m2_t _tmp2 = vfcvt_f_x_v_f16m2(_tmp1, vl);
                    _acc = vfadd_vv_f16m2(_acc, _tmp2, vl);
                    vse16_v_f16m2(acc_buf + c, _acc, vl);
                    c += vl;
                }
            }
        }

        c = 0;
        while (c < in_c) {
            vl = vsetvl_e16m2(in_c - c);
            vfloat16m2_t _acc = vle16_v_f16m2(acc_buf + c, vl);
            _acc = vfmul_vf_f16m2(_acc, ratio, vl);
            vint16m2_t _res0 = vfcvt_x_f_v_i16m2(_acc, vl);
            vint8m1_t _res1 = vnclip_wx_i8m1(_res0, 0, vl);
            _res1 = vadd_vx_i8m1(_res1, z2, vl);
            vse8_v_i8m1(output_data + c, _res1, vl);
            c += vl;
        }
        output_data += in_c;
    }

    shl_mem_free(acc_buf);
    return CSINN_TRUE;
}
