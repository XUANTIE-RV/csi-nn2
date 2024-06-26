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

#include "shl_gref.h"

int shl_gref_space_to_batch_nd(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_space_to_batch_nd_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_SPACE_TO_BATCH_ND, params);
    return CSINN_TRUE;
}

/* input_shape = [batch] + spatial_shape + remaining_shape */
int shl_gref_space_to_batch_nd_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_space_to_batch_nd_params *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    int32_t block_size = 1;
    for (int i = 0; i < params->spatial_dim_cnt; i++) {
        block_size *= params->block_shape[i];
    }

    output->dim_count = input->dim_count;
    output->dim[0] = input->dim[0] * block_size;
    for (int i = 1; i < output->dim_count; i++) {
        int32_t *paddings = params->paddings + i - 1;
        if (i <= params->spatial_dim_cnt) {
            output->dim[i] = (input->dim[i] + params->paddings[(i - 1) * 2] +
                              params->paddings[(i - 1) * 2 + 1]) /
                             params->block_shape[i - 1];
        } else {
            output->dim[i] = input->dim[i];
        }
    }

    return CSINN_TRUE;
}
