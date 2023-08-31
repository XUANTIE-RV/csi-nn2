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

#include "shl_gref.h"

int shl_gref_one_hot(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_one_hot_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_ONE_HOT, params);
    return CSINN_TRUE;
}

int shl_gref_one_hot_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_one_hot_params *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    output->dim_count = input->dim_count + 1;
    if (params->axis == -1) {
        for (int i = 0; i < input->dim_count; i++) {
            output->dim[i] = input->dim[i];
        }
        output->dim[output->dim_count - 1] = params->depth;
    } else {
        for (int i = 0; i < output->dim_count; i++) {
            if (i < params->axis) {
                output->dim[i] = input->dim[i];
            } else if (i == params->axis) {
                output->dim[i] = 1;
            } else {
                output->dim[i] = input->dim[i - 1];
            }
        }
    }
    return CSINN_TRUE;
}
