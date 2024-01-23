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

int shl_rvv_rms_norm_fp32(struct csinn_tensor *input, struct csinn_tensor *weight,
                          struct csinn_tensor *output, struct csinn_rms_norm_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(input);
    }
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *weight_data = (float *)weight->data;
    float eps = params->epsilon;
    /* support negative axis */
    int axis = params->axis >= 0 ? params->axis : (params->axis + input->dim_count);
    int32_t batches = 1;
    for (int i = 0; i < axis; i++) {
        batches *= input->dim[i];
    }
    int32_t norm_size = 1;
    for (int i = axis; i < input->dim_count; i++) {
        norm_size *= input->dim[i];
    }

    for (int b = 0; b < batches; b++) {
        float *input_ptr = input_data + b * norm_size;
        float *output_ptr = output_data + b * norm_size;

        vfloat32m1_t _sum = vfmv_v_f_f32m1(0.0f, 1);
        int i = 0;
        while (i < norm_size) {
            int vl = vsetvl_e32m2(norm_size - i);
            vfloat32m2_t _x = vle32_v_f32m2(input_ptr + i, vl);
            vfloat32m2_t _x2 = vfmul_vv_f32m2(_x, _x, vl);
            _sum = vfredosum_vs_f32m2_f32m1(vundefined_f32m1(), _x2, _sum, vl);
            i += vl;
        }

        float sum = vfmv_f_s_f32m1_f32(_sum);
        float scale = 1.0 / sqrt(sum / norm_size + eps);

        i = 0;
        while (i < norm_size) {
            int vl = vsetvl_e32m2(norm_size - i);
            vfloat32m2_t _x = vle32_v_f32m2(input_ptr + i, vl);
            vfloat32m2_t _w = vle32_v_f32m2(weight_data + i, vl);
            vfloat32m2_t _res = vfmul_vf_f32m2(_x, scale, vl);
            _res = vfmul_vv_f32m2(_res, _w, vl);
            vse32_v_f32m2(output_ptr + i, _res, vl);
            i += vl;
        }
    }

    return CSINN_TRUE;
}
