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
int csinn_where_init(struct csinn_tensor *condition, struct csinn_tensor *x, struct csinn_tensor *y,
                     struct csinn_tensor *output, struct csinn_where_params *params)
{
    shl_op_callback_map(&params->base, CSINN_OP_WHERE, output->dtype);
    int (*func)() = shl_get_init_cb(&params->base);
    if (func != NULL) {
        func(condition, x, y, output, params);
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
int csinn_where(struct csinn_tensor *condition, struct csinn_tensor *x, struct csinn_tensor *y,
                struct csinn_tensor *output, struct csinn_where_params *params)
{
    SHL_DEBUG_CALL(shl_where_debug_info(condition, x, y, output, params, __func__));
    int (*func)() = shl_get_p0_cb(&params->base);
    if (func != NULL) {
        func(condition, x, y, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
/**
 * @}
 */