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

int csi_fsmn_init(struct csi_tensor *frame,
                  struct csi_tensor *l_filter,
                  struct csi_tensor *r_filter,
                  struct csi_tensor *frame_sequence,
                  struct csi_tensor *frame_counter,
                  struct csi_tensor *output,
                  struct fsmn_params *params)
{
    params->base.bc = csi_bc_map(params->base.api, params->base.run_mode, CSINN_OP_FSMN, frame->dtype);
    if (params->base.bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_fsmn(struct csi_tensor *frame,
             struct csi_tensor *l_filter,
             struct csi_tensor *r_filter,
             struct csi_tensor *frame_sequence,
             struct csi_tensor *frame_counter,
             struct csi_tensor *output,
             struct fsmn_params *params)
{
    CSI_DEBUG_CALL(csi_fsmn_debug_info(frame, l_filter, r_filter, frame_sequence, frame_counter, output, params, __func__));
    if (params->base.bc != NULL) {
        params->base.bc(frame, l_filter, r_filter, frame_sequence, frame_counter, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}