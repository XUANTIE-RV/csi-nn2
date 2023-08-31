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

// TODO: support nwc layout
int shl_rvv_conv1d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv1d_params *params)
{
    int32_t out_c = kernel->dim[0];
    int32_t in_c = kernel->dim[1];
    int32_t kernel_w = kernel->dim[2];
    int32_t stride_w = params->stride_width;
    int32_t dilation_w = params->dilation_width;
    int32_t group = params->group;
    struct csinn_callback *cb = params->base.cb;

    if (params->base.quant_type != CSINN_QUANT_INT8_ASYM_W_SYM) {
        cb->exec = shl_ref_conv1d_quant;
        return CSINN_TRUE;
    }

    if (group == 1) {
        if (kernel_w == 1 && stride_w == 1 && dilation_w == 1) {
            // enable fuse zeropoint to bias for gemm
            if (CSINN_TRUE) {
                int32_t *bias_data = (int32_t *)bias->data;
                int8_t *kernel_data = (int8_t *)kernel->data;
                int32_t input_zp = input->qinfo->zero_point;

                if (bias_data == NULL) {
                    // XXX: memory leak
                    bias_data = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));
                    bias->data = bias_data;
                }
                int kernel_inner = in_c * kernel_w;
                for (int oc = 0; oc < out_c; oc++) {
                    int32_t tmp = 0;
                    for (int j = 0; j < kernel_inner; j++) {
                        tmp += kernel_data[oc * kernel_inner + j] * input_zp;
                    }
                    bias_data[oc] -= tmp;
                }
            }
            shl_rvv_conv1d_gemm_reorder_kernel_int8(kernel, params);
            cb->exec = shl_rvv_conv1d_gemm_int8;
        } else {
            cb->exec = shl_ref_conv1d_quant;
        }
        for (int i = 0; i < kernel->quant_channel; i++) {
            float real_scale = input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
            shl_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                    &(kernel->qinfo[i].shift));
        }
    }
    // dwconv1d
    else if (group == input->dim[1] && kernel->dim[1] == 1) {
        const int32_t depth_multiplier = out_c / in_c;
        assert(in_c * depth_multiplier ==
               out_c);  // The input and output channels are equal for dw convolution

        if (bias->data != NULL && bias->dim_count != 0) {
            int32_t *bias_data = (int32_t *)bias->data;
            int8_t *kernel_data = (int8_t *)kernel->data;
            int32_t input_zp = input->qinfo->zero_point;

            // fuse zeropoint to bias
            for (int ic = 0; ic < in_c; ic++) {
                for (int m = 0; m < depth_multiplier; m++) {
                    int oc = m + ic * depth_multiplier;
                    int32_t tmp = 0;
                    for (int j = 0; j < kernel_w; j++) {
                        tmp += kernel_data[oc * kernel_w + j] * input_zp;
                    }
                    bias_data[oc] -= tmp;
                }
            }

            for (int i = 0; i < kernel->quant_channel; i++) {
                float real_scale =
                    input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
                shl_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                        &(kernel->qinfo[i].shift));
            }

            cb->exec = shl_rvv_dwconv1d_int8;
        } else {
            cb->exec = shl_ref_conv1d_quant;
        }
    }
    // group conv1d
    else {
        cb->exec = shl_ref_conv1d_quant;
    }
    return CSINN_TRUE;
}
