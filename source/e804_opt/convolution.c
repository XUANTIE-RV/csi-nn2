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

/* CSI-NN2 version 1.12.x */

#include "csi_e804.h"


static int csi_e804_conv2d_q7(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct csi_tensor *kernel,
                              struct csi_tensor *bias,
                              struct conv2d_params *params)
{
    q7_t *input_data    = (q7_t *)input->data;
    q7_t *kernel_data   = (q7_t *)kernel->data;
    q7_t *bias_data     = (q7_t *)bias->data;
    q7_t *output_data   = (q7_t *)output->data;

    uint16_t batch = input->dim[0]; // batch = 1
    uint16_t in_h = input->dim[1];
    uint16_t in_w = input->dim[2];
    uint16_t in_c = input->dim[3];

    uint16_t out_h = output->dim[1];
    uint16_t out_w = output->dim[2];
    uint16_t out_c = output->dim[3];

    // kernel_layout: OIHW
    uint16_t kernel_h = kernel->dim[2];
    uint16_t kernel_w = kernel->dim[3];

    uint16_t stride_h = params->stride_height;
    uint16_t stride_w = params->stride_width;

    uint16_t pad_x = params->pad_left;
    uint16_t pad_y = params->pad_top;

    q15_t buffer_tmp[2 * in_c * kernel_h * kernel_w];  // buffer_size = in_c * kernel_size * kernel_size

    if ( (in_c % 4 == 0) && (out_c % 2 == 0) ) {
        if ( (kernel_h == 1) && (kernel_w == 1) ) {
            csky_dsp2_convolve_1x1_HWC_q7_fast(input_data, in_w, in_h, in_c, kernel_data, out_c,
                                               bias_data, bias->qinfo->shift, output->qinfo->shift, output_data,
                                               out_w, out_h, buffer_tmp);
        } else {
            csky_dsp2_convolve_HWC_q7_basic(input_data, in_h, in_c, kernel_data, out_c, kernel_h,
                                            pad_y, stride_h, bias_data, bias->qinfo->shift, output->qinfo->shift,
                                            output_data, out_h, buffer_tmp);
        }
    } else if (in_c == 3) {
        csky_dsp2_convolve_HWC_q7_RGB(input_data, in_h, kernel_data, out_c, kernel_h,
                                      pad_y, stride_h, bias_data, bias->qinfo->shift, output->qinfo->shift,
                                      output_data, out_h, buffer_tmp);
    } else {
        csky_dsp2_convolve_HWC_q7_basic(input_data, in_h, in_c, kernel_data, out_c, kernel_h,
                                        pad_y, stride_h, bias_data, bias->qinfo->shift, output->qinfo->shift,
                                        output_data, out_h, buffer_tmp);
    }
    return CSINN_TRUE;
}

static int csi_e804_conv2d_q15(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct csi_tensor *kernel,
                               struct csi_tensor *bias,
                               struct conv2d_params *params)
{
    q15_t *input_data   = (q15_t *)input->data;
    q15_t *kernel_data  = (q15_t *)kernel->data;
    q15_t *bias_data    = (q15_t *)bias->data;
    q15_t *output_data  = (q15_t *)output->data;

    uint16_t batch = input->dim[0];
    uint16_t in_hw = input->dim[1];     // e.g. in_hw = input->dim[2];
    uint16_t in_c  = input->dim[3];

    uint16_t out_hw = output->dim[1];   // e.g. out_hw = output->dim[2]
    uint16_t out_c = output->dim[3];

    uint16_t kernel_size = kernel->dim[2];      // e.g. kernel_size = kernel->dim[3];
    uint16_t stride = params->stride_height;    // e.g. stride = params->stride_width
    uint16_t padding = params->pad_top;         // e.g. padding = params->down = params->left = params->right

    q15_t buffer_tmp[in_c * kernel_size * kernel_size];  // buffer_size = in_c * kernel_size * kernel_size

    csky_dsp2_convolve_HWC_q15_basic(input_data, in_hw, in_c, kernel_data, out_c,
                                      kernel_size, padding, stride, bias_data, bias->qinfo->shift,
                                      output->qinfo->shift, output_data, out_hw, buffer_tmp);

    return CSINN_TRUE;
}

static int csi_e804_depthwise_conv2d_q7(struct csi_tensor *input,
                                        struct csi_tensor *output,
                                        struct csi_tensor *kernel,
                                        struct csi_tensor *bias,
                                        struct conv2d_params *params)
{
    q7_t *input_data    = (q7_t *)input->data;
    q7_t *kernel_data   = (q7_t *)kernel->data;
    q7_t *bias_data     = (q7_t *)bias->data;
    q7_t *output_data   = (q7_t *)output->data;

    uint16_t batch = input->dim[0];
    uint16_t in_hw = input->dim[1];     // e.g. in_hw = input->dim[2];
    uint16_t in_c  = input->dim[3];

    uint16_t out_hw = output->dim[1];   // e.g. out_hw = output->dim[2]
    uint16_t out_c = output->dim[3];

    uint16_t kernel_size = kernel->dim[2];      // e.g. kernel_size = kernel->dim[3];
    uint16_t stride = params->stride_height;    // e.g. stride = params->stride_width
    uint16_t padding = params->pad_top;         // e.g. padding = params->down = params->left = params->right

    q15_t buffer_tmp[2 * in_c * kernel_size * kernel_size];  // buffer_size = in_c * kernel_size * kernel_size

    csky_dsp2_depthwise_separable_conv_HWC_q7(input_data, in_hw, in_c, kernel_data, out_c, kernel_size,
                                              padding, stride, bias_data, bias->qinfo->shift, output->qinfo->shift,
                                              output_data, out_hw, buffer_tmp);

    return CSINN_TRUE;
}

int csi_e804_conv2d_init_q7(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *kernel,
                            struct csi_tensor *bias,
                            struct conv2d_params *params)
{
    uint8_t flag = 0;
    if ( (params->pad_top != params->pad_down) || (params->pad_left != params->pad_right) ) {
        flag |= 0x01;
    }
    if ( (input->dim[1] != input->dim[2]) || (kernel->dim[2] != kernel->dim[3]) || 
         (params->pad_left != params->pad_top) || (params->stride_height != params->stride_width) ) {
        if ( (input->dim[3] % 4 != 0) || (output->dim[3] % 2 != 0) ) {
            flag |= 0x02;
        } else {
            if (kernel->dim[2] != 1 || kernel->dim[3] != 1) {
                flag |= 0x04;
            }
        }
    }
    if (flag > 0) {
        csi_debug_warning("conv2d q7 is not optimized to achieve under this condition on e804, call reference func replaced.\n");
        params->base.bc = csi_ref_conv2d_quant;
    } else {
        params->base.bc = csi_e804_conv2d_q7;
    }
    return CSINN_TRUE;
}

int csi_e804_conv2d_init_q15(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct csi_tensor *kernel,
                             struct csi_tensor *bias,
                             struct conv2d_params *params)
{
    uint8_t flag = 0;
    if ( (params->pad_top != params->pad_down) || (params->pad_left != params->pad_right) ||
         (params->pad_top != params->pad_left) ) {
        flag |= 0x01;
    }
    if (input->dim[1] != input->dim[2]) {
        flag |= 0x02;   
    }
    if (kernel->dim[2] != kernel->dim[3]) {
        flag |= 0x04;
    }
    if (params->stride_height != params->stride_width) {
        flag |= 0x08;
    }
    if (flag > 0) {
        csi_debug_warning("conv2d q15 is not optimized to achieve under this condition on e804, call reference func replaced.\n");
        params->base.bc = csi_ref_conv2d_quant;
    } else {
        params->base.bc = csi_e804_conv2d_q15;
    }
    return CSINN_TRUE;
}

int csi_e804_depthwise_conv2d_init_q7(struct csi_tensor *input,
                                      struct csi_tensor *output,
                                      struct csi_tensor *kernel,
                                      struct csi_tensor *bias,
                                      struct conv2d_params *params)
{

    uint8_t flag = 0;
    if ( (params->pad_top != params->pad_down) || (params->pad_left != params->pad_right) ||
         (params->pad_top != params->pad_left) ) {
        flag |= 0x01;
    }
    if (input->dim[1] != input->dim[2]) {
        flag |= 0x02;   
    }
    if (kernel->dim[2] != kernel->dim[3]) {
        flag |= 0x04;
    }
    if (params->stride_height != params->stride_width) {
        flag |= 0x08;
    }
    if (flag > 0) {
        params->base.bc = csi_ref_depthwise_conv2d_quant;
        csi_debug_warning("depthwise_conv2d q7 is not optimized to achieve under this condition on e804, call reference func replaced.\n");

    } else {
        params->base.bc = csi_e804_depthwise_conv2d_q7;
    }
    return CSINN_TRUE;
}
