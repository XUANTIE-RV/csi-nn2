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

int shl_gref_gather(struct csinn_tensor *input, struct csinn_tensor *indices,
                    struct csinn_tensor *output, struct csinn_gather_params *params)
{
    shl_gref_diso_op(input, indices, output, CSINN_OP_GATHER, params);
    return CSINN_TRUE;
}

int shl_gref_gather_infer_shape(struct csinn_tensor *input, struct csinn_tensor *indices,
                                struct csinn_tensor *output, struct csinn_gather_params *params)
{
    int32_t axis = params->axis;
    int32_t indices_dim_count = indices->dim_count;
    // if indices is a single number
    if (indices_dim_count == 1 && indices->dim[0] == 1) {
        indices_dim_count = 0;
    }
    output->dim_count = input->dim_count + indices_dim_count - 1;
    int j = 0;
    for (int i = 0; i < axis; i++) {
        output->dim[j] = input->dim[i];
        j++;
    }
    for (int i = 0; i < indices_dim_count; i++) {
        output->dim[j] = indices->dim[i];
        j++;
    }
    for (int i = axis + 1; i < input->dim_count; i++) {
        output->dim[j] = input->dim[i];
        j++;
    }
    return CSINN_TRUE;
}
