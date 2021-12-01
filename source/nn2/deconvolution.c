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

/* CSI-NN2 version 1.10.x */

#include "csi_nn.h"

int csi_deconv2d_init(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct csi_tensor *kernel,
                      struct csi_tensor *bias,
                      struct conv2d_params *params)
{
    if (params->group == 1) {
        params->base.bc = csi_bc_map(params->base.api, params->base.run_mode, CSINN_OP_DECONV2D, input->dtype);
        if (params->base.bc == NULL) {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else if ( (params->group == output->dim[1] && params->base.layout == CSINN_LAYOUT_NCHW) ||
                (params->group == output->dim[3] && params->base.layout == CSINN_LAYOUT_NHWC) ) {
        params->base.bc = csi_bc_map(params->base.api, params->base.run_mode, CSINN_OP_DEPTHWISE_DECONV2D, input->dtype);
        if (params->base.bc == NULL) {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else {
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}

int csi_deconv2d(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct csi_tensor *kernel,
                 struct csi_tensor *bias,
                 struct conv2d_params *params)
{
    CSI_DEBUG_CALL(csi_conv2d_debug_info(input, output, kernel, bias, params, __func__));
    if (params->base.bc != NULL) {
        params->base.bc(input, output, kernel, bias, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
