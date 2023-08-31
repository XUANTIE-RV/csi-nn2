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

#include "rvv/rvv.h"

/*************************************************************
    note: VLEN = 128/256
*************************************************************/
int shl_rvv_global_maxpool2d_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_pool_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int batch = input->dim[0];
    int in_h = input->dim[1];
    int in_w = input->dim[2];
    int in_c = input->dim[3];
    int vl;
    int c;

    __fp16 *max_buf = (__fp16 *)shl_mem_alloc(in_c * sizeof(__fp16));

    for (int b = 0; b < batch; b++) {
        c = 0;
        while (c < in_c) {
            vl = vsetvl_e16m1(in_c - c);
            vfloat16m1_t _tmp = vfmv_v_f_f16m1(-__FLT_MAX__, vl);
            vse16_v_f16m1(max_buf + c, _tmp, vl);
            c += vl;
        }

        const __fp16 *src = (__fp16 *)input_data + b * in_h * in_w * in_c;
        for (int h = 0; h < in_h; h++) {
            for (int w = 0; w < in_w; w++) {
                const __fp16 *in_ptr = src + (h * in_w + w) * in_c;
                c = 0;
                while (c < in_c) {
                    vl = vsetvl_e16m1(in_c - c);
                    vfloat16m1_t _max = vle16_v_f16m1(max_buf + c, vl);
                    _max = vfmax_vv_f16m1(_max, vle16_v_f16m1(in_ptr + c, vl), vl);
                    vse16_v_f16m1(max_buf + c, _max, vl);
                    c += vl;
                }
            }
        }

        c = 0;
        while (c < in_c) {
            vl = vsetvl_e16m1(in_c - c);
            vfloat16m1_t _max = vle16_v_f16m1(max_buf + c, vl);
            vse16_v_f16m1(output_data + c, _max, vl);
            c += vl;
        }
        output_data += in_c;
    }

    shl_mem_free(max_buf);
    // requantize
    shl_rvv_siso_op_requantize_fp16(input, output);
    return CSINN_TRUE;
}
