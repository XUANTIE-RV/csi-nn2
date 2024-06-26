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

static int rms_norm_fp16(struct csinn_tensor *input, struct csinn_tensor *weight,
                         struct csinn_tensor *output, struct csinn_rms_norm_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input);
    }
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weight_data = (__fp16 *)weight->data;
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
        __fp16 *input_ptr = input_data + b * norm_size;
        __fp16 *output_ptr = output_data + b * norm_size;

        vfloat32m1_t _sum = vfmv_v_f_f32m1(0.0f, 1);
        int i = 0;
        while (i < norm_size) {
            int vl = vsetvl_e16m2(norm_size - i);
            vfloat16m2_t _x = vle16_v_f16m2(input_ptr + i, vl);
            vfloat32m4_t _x_f32 = vfwcvt_f_f_v_f32m4(_x, vl);
            vfloat32m4_t _x2 = vfmul_vv_f32m4(_x_f32, _x_f32, vl);
            _sum = vfredosum_vs_f32m4_f32m1(vundefined_f32m1(), _x2, _sum, vl);
            i += vl;
        }

        float sum = vfmv_f_s_f32m1_f32(_sum);
        float scale = 1.0 / sqrt(sum / norm_size + eps);

        i = 0;
        while (i < norm_size) {
            int vl = vsetvl_e16m2(norm_size - i);
            vfloat16m2_t _x = vle16_v_f16m2(input_ptr + i, vl);
            vfloat32m4_t _x_f32 = vfwcvt_f_f_v_f32m4(_x, vl);
            vfloat16m2_t _w = vle16_v_f16m2(weight_data + i, vl);
            vfloat32m4_t _w_f32 = vfwcvt_f_f_v_f32m4(_w, vl);
            vfloat32m4_t _res_f32 = vfmul_vf_f32m4(_x_f32, scale, vl);
            _res_f32 = vfmul_vv_f32m4(_res_f32, _w_f32, vl);
            vfloat16m2_t _res = vfncvt_f_f_w_f16m2(_res_f32, vl);
            vse16_v_f16m2(output_ptr + i, _res, vl);
            i += vl;
        }
    }

    return CSINN_TRUE;
}

int rms_norm_fp16_w_fp32(struct csinn_tensor *input, struct csinn_tensor *weight,
                         struct csinn_tensor *output, struct csinn_rms_norm_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input);
    }
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
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
        __fp16 *input_ptr = input_data + b * norm_size;
        __fp16 *output_ptr = output_data + b * norm_size;

        vfloat32m1_t _sum = vfmv_v_f_f32m1(0.0f, 1);
        int i = 0;
        while (i < norm_size) {
            int vl = vsetvl_e16m2(norm_size - i);
            vfloat16m2_t _x = vle16_v_f16m2(input_ptr + i, vl);
            vfloat32m4_t _x_f32 = vfwcvt_f_f_v_f32m4(_x, vl);
            vfloat32m4_t _x2 = vfmul_vv_f32m4(_x_f32, _x_f32, vl);
            _sum = vfredosum_vs_f32m4_f32m1(vundefined_f32m1(), _x2, _sum, vl);
            i += vl;
        }

        float sum = vfmv_f_s_f32m1_f32(_sum);
        float scale = 1.0 / sqrt(sum / norm_size + eps);

        i = 0;
        while (i < norm_size) {
            int vl = vsetvl_e16m2(norm_size - i);
            vfloat16m2_t _x = vle16_v_f16m2(input_ptr + i, vl);
            vfloat32m4_t _x_f32 = vfwcvt_f_f_v_f32m4(_x, vl);
            vfloat32m4_t _w_f32 = vle32_v_f32m4(weight_data + i, vl);
            vfloat32m4_t _res_f32 = vfmul_vf_f32m4(_x_f32, scale, vl);
            _res_f32 = vfmul_vv_f32m4(_res_f32, _w_f32, vl);
            vfloat16m2_t _res = vfncvt_f_f_w_f16m2(_res_f32, vl);
            vse16_v_f16m2(output_ptr + i, _res, vl);
            i += vl;
        }
    }

    return CSINN_TRUE;
}

int shl_rvv_rms_norm_fp16(struct csinn_tensor *input, struct csinn_tensor *weight,
                          struct csinn_tensor *output, struct csinn_rms_norm_params *params)
{
    if (output->dtype == CSINN_DTYPE_FLOAT16) {
        if (weight->dtype == CSINN_DTYPE_FLOAT16) {
            return rms_norm_fp16(input, weight, output, params);
        } else if (weight->dtype == CSINN_DTYPE_FLOAT32) {
            return rms_norm_fp16_w_fp32(input, weight, output, params);
        }
    }

    return shl_ref_rms_norm_quant(input, weight, output, params);
}
