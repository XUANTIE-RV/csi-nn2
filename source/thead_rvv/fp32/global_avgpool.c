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
int shl_rvv_global_avgpool2d_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int in_hw = in_h * in_w;
    int vl;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            vfloat32m1_t _res = vfmv_s_f_f32m1(vundefined_f32m1(), 0.0f, 4);
            int size = in_hw;
            while (size > 0) {
                vl = vsetvl_e32m2(size);
                vfloat32m2_t _input = vle32_v_f32m2(input_data, vl);
                _res = vfredusum_vs_f32m2_f32m1(vundefined_f32m1(), _input, _res, vl);
                input_data += vl;
                size -= vl;
            }
            float avg = vfmv_f_s_f32m1_f32(_res) / in_hw;
            *output_data++ = avg;
        }
    }
    return CSINN_TRUE;
}
