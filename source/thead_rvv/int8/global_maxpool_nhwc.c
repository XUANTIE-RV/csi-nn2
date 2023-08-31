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

int shl_rvv_global_maxpool2d_nhwc_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_pool_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    int batch = input->dim[0];
    int in_h = input->dim[1];
    int in_w = input->dim[2];
    int in_c = input->dim[3];
    int vl;
    int c;

    int8_t *max_buf = (int8_t *)shl_mem_alloc(in_c * sizeof(int8_t));

    for (int b = 0; b < batch; b++) {
        c = 0;
        while (c < in_c) {
            vl = vsetvl_e8m1(in_c - c);
            vint8m1_t _tmp = vmv_v_x_i8m1(-128, vl);
            vse8_v_i8m1(max_buf + c, _tmp, vl);
            c += vl;
        }

        const int8_t *src = (int8_t *)input_data + b * in_h * in_w * in_c;
        for (int h = 0; h < in_h; h++) {
            for (int w = 0; w < in_w; w++) {
                const int8_t *in_ptr = src + (h * in_w + w) * in_c;
                c = 0;
                while (c < in_c) {
                    vl = vsetvl_e8m1(in_c - c);
                    vint8m1_t _max = vle8_v_i8m1(max_buf + c, vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(in_ptr + c, vl), vl);
                    vse8_v_i8m1(max_buf + c, _max, vl);
                    c += vl;
                }
            }
        }

        c = 0;
        while (c < in_c) {
            vl = vsetvl_e8m1(in_c - c);
            vint8m1_t _max = vle8_v_i8m1(max_buf + c, vl);
            vse8_v_i8m1(output_data + c, _max, vl);
            c += vl;
        }
        output_data += in_c;
    }

    shl_mem_free(max_buf);
    return CSINN_TRUE;
}