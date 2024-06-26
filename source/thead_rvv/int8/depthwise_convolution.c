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

int shl_rvv_depthwise_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params)
{
    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];
    int32_t out_c = output->dim[1];
    int32_t kernel_h = kernel->dim[2];
    int32_t kernel_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    struct csinn_callback *cb = params->base.cb;

    if (params->base.quant_type != CSINN_QUANT_INT8_ASYM_W_SYM) {
        cb->exec = shl_ref_depthwise_conv2d_quant;
        return CSINN_TRUE;
    }
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int in_elempack = 1;
    int out_elempack = 1;
    struct csinn_session *sess = params->base.sess;
    bool binary_model_op_init = shl_rvv_get_binary_model_op_init(sess);
    if (sess->base_run_mode == CSINN_RM_CPU_GRAPH) {
        struct shl_rvv_option *option = shl_rvv_get_graph_option(sess);
        if (option && option->use_packn_layout) {
            in_elempack = in_c % packn == 0 ? packn : 1;
            out_elempack = out_c % packn == 0 ? packn : 1;
        }
        /* first layer do not convert input layout */
        if (shl_is_first_layer_input(input, sess)) {
            in_elempack = 1;
            out_elempack = 1;  // dwconv2d out_channel pack is same as in_channel
        }
    } else if (sess->base_run_mode == CSINN_RM_LAYER) {
        in_elempack = in_c % packn == 0 ? packn : 1;
        out_elempack = out_c % packn == 0 ? packn : 1;
    }

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
                tmp += kernel_data[oc * kernel_inner + j] * input_zp;
            }
            bias_data[oc] -= tmp;
        }
    }

    if (in_elempack % packn == 0 && out_elempack % packn == 0) {
        if (!binary_model_op_init) {
            shl_rvv_dwconv_reorder_kernel_packn_int8(kernel, params);
        }
        if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
            cb->exec = shl_rvv_dwconv3x3s1_packn_int8;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
            cb->exec = shl_rvv_dwconv3x3s2_packn_int8;
        } else {
            cb->exec = shl_rvv_dwconv_packn_int8;
        }
    }

    if (in_elempack % packn != 0 && out_elempack % packn != 0) {
        if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
            cb->exec = shl_rvv_dwconv3x3s1_int8;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
            cb->exec = shl_rvv_dwconv3x3s2_int8;
        } else {
            cb->exec = shl_ref_depthwise_conv2d_quant;
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
