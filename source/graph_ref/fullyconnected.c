/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#include "shl_gref.h"

int shl_gref_fullyconnected(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *weight, struct csinn_tensor *bias,
                            struct csinn_fc_params *params)
{
    shl_gref_sidcso_op(input, output, weight, bias, CSINN_OP_FULLYCONNECTED, params);
    return CSINN_TRUE;
}

int shl_gref_fullyconnected_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *weights, struct csinn_tensor *bias,
                                        struct csinn_fc_params *params)
{
    output->dim_count = 2;
    output->dim[0] = input->dim[0];
    output->dim[1] = weights->dim[0];
    return CSINN_TRUE;
}
