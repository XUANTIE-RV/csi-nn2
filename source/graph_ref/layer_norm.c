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

#include "csi_gref.h"

int csi_gref_layer_norm(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *gamma,
                        struct csi_tensor *beta,
                        struct layer_norm_params *params)
{
    csi_gref_sidcso_op(input, output, gamma, beta, CSINN_OP_LAYER_NORM, params);
    return CSINN_TRUE;
}