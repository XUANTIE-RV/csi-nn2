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

#include "csi_ref.h"

int csi_ref_broadcast_to_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct broadcast_to_params *params)
{
    return csi_ref_broadcast_to_shape_f32(input, output, params->shape, params->shape_count);
}

int csi_ref_broadcast_to_quant(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct broadcast_to_params *params)
{
    return csi_ref_broadcast_to_shape_quant(input, output, params->shape, params->shape_count);
}
