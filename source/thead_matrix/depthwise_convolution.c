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

#include "rvm/rvm.h"

int shl_rvm_depthwise_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    cb->exec = shl_rvv_dwconv_nhwc_fp16;
    return CSINN_TRUE;
}

int shl_rvm_depthwise_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params)
{
    int32_t in_c = input->dim[3];
    int32_t out_c = output->dim[3];
    int32_t kernel_h = kernel->dim[1];
    int32_t kernel_w = kernel->dim[2];

    struct csinn_callback *cb = params->base.cb;
    cb->exec = shl_rvv_dwconv_nhwc_int8;

    // enable fuse zeropoint to bias
    if (!params->conv_extra.fuse_zp2bias) {
        params->conv_extra.fuse_zp2bias = true;
        int32_t *bias_data = (int32_t *)bias->data;
        int8_t *kernel_data = (int8_t *)kernel->data;
        int32_t input_zp = input->qinfo->zero_point;

        if (bias_data == NULL) {
            // XXX: memory leak
            bias_data = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));
            bias->data = bias_data;
        }
        int kernel_inner = 1 * kernel_h * kernel_w;
        for (int oc = 0; oc < out_c; oc++) {
            int32_t tmp = 0;
            for (int j = 0; j < kernel_inner; j++) {
                tmp += kernel_data[out_c * j + oc] * input_zp;
            }
            bias_data[oc] -= tmp;
        }
    }

    // support channel quantization
    for (int i = 0; i < kernel->quant_channel; i++) {
        float real_scale = input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
        shl_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                &(kernel->qinfo[i].shift));
    }
    return CSINN_TRUE;
}
