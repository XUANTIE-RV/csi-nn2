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

#include "csi_ref_i805.h"


/*
    constraint: 1.input tensor layout: NHWC
                2. pad_left = pad_right; pad_top = pad_down
    FIXME: count_include_pad
*/
static int csi_ref_i805_avgpool2d_q7(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct pool_params *params)
{
    q7_t *input_data = (q7_t *)input->data;
    q7_t *output_data = (q7_t *)output->data;

    uint16_t batch = input->dim[0];
    uint16_t in_h = input->dim[1];
    uint16_t in_w = input->dim[2];
    uint16_t in_c = input->dim[3];

    uint16_t out_h = output->dim[1];
    uint16_t out_w = output->dim[2];

    uint16_t kernel_h = params->filter_height;
    uint16_t kernel_w = params->filter_width;

    uint16_t stride_h = params->stride_height;
    uint16_t stride_w = params->stride_width;

    uint16_t pad_x = params->pad_left;   // i.e. pad_x = params->pad_right
    uint16_t pad_y = params->pad_top;    // i.e. pad_y = params->pad_down

    q7_t buffer_tmp[out_h * out_w * in_c];  // buffer_size = out_h * out_w * channel

    if ( (in_h == in_w) && (kernel_h == kernel_w) && (pad_x == pad_y) && (stride_h == stride_w) ) {
        csi_avepool_q7_HWC(input_data, in_h, in_c, kernel_h, pad_y, stride_h, out_h,
                            buffer_tmp, output_data);
    } else {
        csi_avepool_q7_HWC_nonsquare(input_data, in_w, in_h, in_c, kernel_w, kernel_h,
                                     pad_x, pad_y, stride_w, stride_h, out_w, out_h,
                                     buffer_tmp, output_data, output->qinfo->shift);
    }
    return CSINN_TRUE;
}

int csi_ref_i805_avgpool2d_init_q7(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct pool_params *params)
{
    if ( (params->pad_top != params->pad_down) || (params->pad_left != params->pad_right) ) {
        csi_debug_warning("avgpool q7 unsupport asymmetric padddings on ref_i805, call reference func replaced.\n");
        params->base.bc = csi_ref_avgpool2d_quant;    // FIXME: csi_ref_avgpool2d_quant may be not applicable to i805 
    } else {
        params->base.bc = csi_ref_i805_avgpool2d_q7;
    }
    return CSINN_TRUE;
}        
