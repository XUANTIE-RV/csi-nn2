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

int shl_rvv_rms_norm_int8(struct csinn_tensor *input, struct csinn_tensor *weight,
                          struct csinn_tensor *output, struct csinn_rms_norm_params *params)
{
    struct csinn_tensor *float_input = shl_rvv_tensor_transform_f32(input);
    struct csinn_tensor *float_output = shl_rvv_tensor_transform_f32(output);
    struct csinn_tensor *float_weight = shl_rvv_tensor_transform_f32(weight);

    if (float_input == NULL) {
        shl_debug_warning(
            "shl_rvv_tensor_transform_f32 is not optimized to achieve under this condition on RVV, "
            "call reference func replaced.\n");
        float_input = shl_ref_tensor_transform_f32(input);
    }
    if (float_output == NULL) {
        shl_debug_warning(
            "shl_rvv_tensor_transform_f32 is not optimized to achieve under this condition on RVV, "
            "call reference func replaced.\n");
        float_output = shl_ref_tensor_transform_f32(output);
    }
    if (float_weight == NULL) {
        shl_debug_warning(
            "shl_rvv_tensor_transform_f32 is not optimized to achieve under this condition on RVV, "
            "call reference func replaced.\n");
        float_weight = shl_ref_tensor_transform_f32(weight);
    }

    int ret = shl_rvv_rms_norm_fp32(float_input, float_weight, float_output, params);

    if (shl_rvv_tensor_data_convert(float_output, output) != CSINN_TRUE) {
        shl_debug_warning(
            "shl_rvv_tensor_data_convert is not optimized to achieve under this condition on RVV, "
            "call reference func replaced.\n");
        csinn_tensor_data_convert(output, float_output);
    }

    shl_ref_tensor_transform_free_f32(float_input);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_weight);

    return ret;
}
