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

/* SHL version 2.1.x */

#include "csi_nn.h"
#include "shl_utils.h"

int csinn_segment_mean_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_segment_params *params)
{
    if (params->unsorted == CSINN_TRUE) {
        shl_op_callback_map(&params->base, CSINN_OP_UNSORTED_SEGMENT_MEAN, input0->dtype);
    } else {
        shl_op_callback_map(&params->base, CSINN_OP_SEGMENT_MEAN, input0->dtype);
    }
    int (*func)() = shl_get_init_cb(&params->base);
    if (func != NULL) {
        func(input0, input1, output, params);
    }
    return CSINN_TRUE;
}

int csinn_segment_mean(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_segment_params *params)
{
    SHL_DEBUG_CALL(shl_segment_debug_info(input0, input1, output, params, __func__));
    int (*func)() = shl_get_p0_cb(&params->base);
    if (func != NULL) {
        func(input0, input1, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
