/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* CSI-NN2 version 1.13.x */

#include "csi_thead_rvv.h"

int csi_nn_rvv_avgpool2d_init(struct csi_tensor *input, struct csi_tensor *output,
                              struct pool_params *params)
{
    int32_t input_h = input->dim[2];
    int32_t input_w = input->dim[3];

    int32_t kernel_h = params->filter_height;
    int32_t kernel_w = params->filter_width;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;
    int32_t pad_top = params->pad_top;
    int32_t pad_down = params->pad_down;

    params->base.bc = NULL;

    // global avgpool2d
    if (input_h == kernel_h && input_w == kernel_w) {
        if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->base.bc = csi_nn_rvv_global_avgpool2d_fp32;
        } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
            params->base.bc = csi_nn_rvv_global_avgpool2d_fp16;
        } else if (input->dtype == CSINN_DTYPE_INT8) {
            params->base.bc = csi_ref_avgpool2d_quant;
        }
        return CSINN_TRUE;
    }

    if (stride_h == 2 && stride_w == 2) {
        if (kernel_h == 2 && kernel_w == 2) {
            if (pad_left == 0 && pad_top == 0) {
                // adjust pad according to ceil_mode (ceil mode on caffe pytorch..)
                if (input_h % 2 == 1 && params->ceil_mode == 1) {
                    if (params->pad_down) params->pad_down++;
                }
                if (input_w % 2 == 1 && params->ceil_mode == 1) {
                    if (params->pad_right) params->pad_right++;
                }
                // end consider ceil_mode 2x2s2p0

                if (input->dtype == CSINN_DTYPE_FLOAT32) {
                    params->base.bc = csi_nn_rvv_avgpool2x2s2_fp32;
                } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
                    params->base.bc = csi_nn_rvv_avgpool2x2s2_fp16;
                }
            } else if (pad_left == 1 && pad_top == 1) {
                if (input->dtype == CSINN_DTYPE_FLOAT32) {
                    params->base.bc = csi_nn_rvv_avgpool2x2s2_p1_fp32;
                } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
                    params->base.bc = csi_nn_rvv_avgpool2x2s2_p1_fp16;
                }
            }
        } else if (kernel_h == 3 && kernel_w == 3) {
            if (pad_left == 0 && pad_top == 0) {
                // adjust pad according to ceil_mode (ceil mode on caffe pytorch..)
                if (input_h % 2 == 0 && params->ceil_mode == 1) {
                    if (params->pad_down)
                        params->pad_down++;  // origin pad_down mast be equal to zero ?
                }
                if (input_w % 2 == 0 && params->ceil_mode == 1) {
                    if (params->pad_right) params->pad_right++;
                }
                // end consider ceil_mode 3x3s2p0

                if (input->dtype == CSINN_DTYPE_FLOAT32) {
                    params->base.bc = csi_nn_rvv_avgpool3x3s2_fp32;
                } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
                    params->base.bc = csi_nn_rvv_avgpool3x3s2_fp16;
                }
            } else if (pad_left == 1 && pad_top == 1) {
                if (input->dtype == CSINN_DTYPE_FLOAT32) {
                    params->base.bc = csi_nn_rvv_avgpool3x3s2_p1_fp32;
                } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
                    params->base.bc = csi_nn_rvv_avgpool3x3s2_p1_fp16;
                }
            }
        }
    } else if (stride_h == 1 && stride_w == 1) {
        if (kernel_h == 3 && kernel_w == 3) {
            if (pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
                if (input->dtype == CSINN_DTYPE_FLOAT32) {
                    params->base.bc = csi_nn_rvv_avgpool3x3s1_p1_fp32;
                } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
                    params->base.bc = csi_nn_rvv_avgpool3x3s1_p1_fp16;
                }
            }
        }
    }

    if (params->base.bc == NULL) {
        csi_debug_warning(
            "avgpool is not optimized to achieve under this condition on RVV, call reference func "
            "replaced.\n");
        if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->base.bc = csi_ref_avgpool2d_f32;
        } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
            params->base.bc = csi_ref_avgpool2d_quant;
        } else if (input->dtype == CSINN_DTYPE_INT8) {
            params->base.bc = csi_ref_avgpool2d_quant;
        }
    }
    return CSINN_TRUE;
}
