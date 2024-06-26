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

#include "e907/e907.h"

int shl_e907_conv2d_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv2d_params *params)
{
    int ret;
    if (params->conv_extra.fuse_zp2bias) {
        struct csinn_tensor *tmp_bias = shl_ref_tensor_transform_f32(bias);
        struct csinn_tensor *tmp_kernel = shl_ref_tensor_transform_f32(kernel);
        float *tmp_bias_data = tmp_bias->data;
        float *tmp_kernel_data = tmp_kernel->data;

        int k_len = kernel->dim[0];
        int k_inner = csinn_tensor_size(kernel) / k_len;
        float sp = input->qinfo->scale * input->qinfo->zero_point;
        for (int i = 0; i < k_len; i++) {
            float t_k = 0;
            for (int j = 0; j < k_inner; j++) {
                int k_idx = i * k_inner + j;
                t_k += tmp_kernel_data[k_idx] * sp;
            }
            tmp_bias_data[i] += t_k;
        }
        shl_ref_tensor_transform_free_f32(tmp_kernel);
        ret =
            shl_ref_conv_callback_base(input, output, kernel, tmp_bias, params, shl_ref_conv2d_f32);
        shl_ref_tensor_transform_free_f32(tmp_bias);
    } else {
        ret = shl_ref_conv_callback_base(input, output, kernel, bias, params, shl_ref_conv2d_f32);
    }
    return ret;
}
