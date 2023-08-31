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

#ifdef SHL_USE_DOT_INT4
int shl_rvv_conv2d_init_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params)
{
    int32_t out_c = kernel->dim[0] / params->group;
    int32_t in_c = kernel->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t kernel_h = kernel->dim[2];
    int32_t kernel_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;
    struct csinn_callback *cb = params->base.cb;

    // xxx: only int4 support nhwc layout now
    if (input->layout == CSINN_LAYOUT_NHWC) {
        out_c = kernel->dim[0];
        in_c = kernel->dim[3];
        in_h = input->dim[1];
        in_w = input->dim[2];
        kernel_h = kernel->dim[1];
        kernel_w = kernel->dim[2];
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
            dilation_w == 1) {
            params->conv_extra.conv_mode = CSINN_GEMM;
            if (input->dtype == CSINN_DTYPE_INT4) {
                params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
                shl_rvv_conv1x1s1_gemm_reorder_kernel_int4(kernel, params);
                // support channel quantization
                for (int i = 0; i < kernel->quant_channel; i++) {
                    float real_scale =
                        input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
                    shl_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                            &(kernel->qinfo[i].shift));
                }
                cb->exec = shl_rvv_conv1x1s1_gemm_int4;
            }
        } else {
            params->conv_extra.conv_mode = CSINN_GEMM;
            if (input->dtype == CSINN_DTYPE_INT4) {
                params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
                shl_rvv_conv_im2col_gemm_reorder_kernel_int4(kernel, params);
                for (int i = 0; i < kernel->quant_channel; i++) {
                    float real_scale =
                        input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
                    shl_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                            &(kernel->qinfo[i].shift));
                }
                cb->exec = shl_rvv_conv_im2col_gemm_int4;
            }
        }
        return CSINN_TRUE;
    }
    return CSINN_FALSE;
}
#endif
