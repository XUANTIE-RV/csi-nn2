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

#include "csi_nn.h"
#include "shl_utils.h"

/**
 * @addtogroup INIT
 * @{
 */
int csinn_conv1d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv1d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NCW) {
        if (params->group == 1) {
            shl_op_callback_map(&params->base, CSINN_OP_CONV1D, input->dtype);
        } else if (params->group == kernel->dim[0] && kernel->dim[1] == 1) {
            shl_op_callback_map(&params->base, CSINN_OP_DEPTHWISE_CONV1D, input->dtype);
        } else {
            shl_op_callback_map(&params->base, CSINN_OP_GROUP_CONV1D, input->dtype);
        }
    } else if (input->layout == CSINN_LAYOUT_NWC) {
        if (params->group == 1) {
            shl_op_callback_map(&params->base, CSINN_OP_CONV1D, input->dtype);
        } else if (params->group == kernel->dim[1] && kernel->dim[0] == 1) {
            shl_op_callback_map(&params->base, CSINN_OP_DEPTHWISE_CONV1D, input->dtype);
        } else {
            shl_op_callback_map(&params->base, CSINN_OP_GROUP_CONV1D, input->dtype);
        }
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }

    int (*func)() = shl_get_init_cb(&params->base);
    if (func != NULL) {
        func(input, output, kernel, bias, params);
    }
    return CSINN_TRUE;
}
/**
 * @}
 */

/**
 * @addtogroup NN
 * @{
 */
int csinn_conv1d(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                 struct csinn_conv1d_params *params)
{
    SHL_DEBUG_CALL(shl_conv1d_debug_info(input, output, kernel, bias, params, __func__));
    int (*func)() = shl_get_p0_cb(&params->base);
    if (func != NULL) {
        func(input, output, kernel, bias, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
/**
 * @}
 */