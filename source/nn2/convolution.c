/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.8.x */

#include "csi_nn.h"

int csi_conv2d_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv2d_params *params)
{
    if (params->base.run_mode != CSINN_RM_CPU_GRAPH) {
        int (*init_func)();
        init_func = csi_init_map(params->base.api, CSINN_OP_CONV2D, input->dtype);
        if (init_func != NULL) {
            return init_func(input, output, kernel, bias, params);
        }
    }

    if (params->base.layout == CSINN_LAYOUT_NCHW) {
        if (params->group == 1) {
            params->base.bc = csi_bc_map(params->base.api, params->base.run_mode, CSINN_OP_CONV2D, input->dtype);
        } else if (params->group == input->dim[1] && kernel->dim[1] == 1) {
            params->base.bc = csi_bc_map(params->base.api, params->base.run_mode, CSINN_OP_DEPTHWISE_CONV2D, input->dtype);
        } else {
            params->base.bc = csi_bc_map(params->base.api, params->base.run_mode, CSINN_OP_GROUP_CONV2D, input->dtype);
        }
        if (params->base.bc == NULL) {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else if (params->base.layout == CSINN_LAYOUT_NHWC) {
        if (params->group == 1) {
            params->base.bc = csi_bc_map(params->base.api, params->base.run_mode, CSINN_OP_CONV2D, input->dtype);
        } else if (params->group == input->dim[3] && kernel->dim[0] == 1) {
            params->base.bc = csi_bc_map(params->base.api, params->base.run_mode, CSINN_OP_DEPTHWISE_CONV2D, input->dtype);
        } else {
            params->base.bc = csi_bc_map(params->base.api, params->base.run_mode, CSINN_OP_GROUP_CONV2D, input->dtype);
        }
        if (params->base.bc == NULL) {
            return CSINN_UNSUPPORT_DTYPE;
        }
    }
    else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int csi_conv2d(struct csi_tensor *input,
               struct csi_tensor *output,
               struct csi_tensor *kernel,
               struct csi_tensor *bias,
               struct conv2d_params *params)
{
    CSI_DEBUG_CALL(csi_conv2d_debug_info(input, output, kernel, bias, params, __func__));
    if (params->base.bc != NULL) {
        if (params->conv_extra.kernel_tm != NULL && params->conv_extra.conv_mode == CSINN_WINOGRAD) {
            params->base.bc(input, output, params->conv_extra.kernel_tm, bias, params);
            free(params->conv_extra.kernel_tm->data);
            free(params->conv_extra.kernel_tm);
        } else {
            params->base.bc(input, output, kernel, bias, params);
        }
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
