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

#include "csi_ref.h"
#include "csi_utils.h"

int csi_ref_shape_i32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct shape_params *params)
{
    int32_t *data = output->data;
    for (int i = 0; i < input->dim_count; i++) {
        data[i] = input->dim[i];
    }
    return CSINN_TRUE;
}

int csi_ref_shape_u8(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct shape_params *params)
{
    uint8_t * data = output->data;
    for (int i = 0; i < input->dim_count; i++) {
        data[i] = input->dim[i];
    }
    return CSINN_TRUE;
}

int csi_ref_shape_i8(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct shape_params *params)
{
    uint8_t * data = output->data;
    for (int i = 0; i < input->dim_count; i++) {
        data[i] = input->dim[i];
    }
    return CSINN_TRUE;
}
