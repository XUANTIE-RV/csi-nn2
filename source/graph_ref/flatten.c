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

int shl_gref_flatten(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_flatten_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_FLATTEN, params);
    return CSINN_TRUE;
}

int shl_gref_flatten_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_flatten_params *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    int in_size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        in_size *= input->dim[i];
    }

    int32_t axis = params->axis < 0 ? params->axis + input->dim_count : params->axis;
    if (axis >= input->dim_count) {
        shl_debug_fatal(
            "flatten axis must less than input dim count, but flatten %s, get axis %d, input dim "
            "count %d",
            params->base.name, params->axis, input->dim_count);
    }

    output->dim_count = 2;
    if (axis == 0) {
        output->dim[0] = 1;
        output->dim[1] = in_size;
    } else {
        int outer_size = 1;
        for (int i = 0; i < axis; i++) {
            outer_size *= input->dim[i];
        }
        output->dim[0] = outer_size;
        output->dim[1] = in_size / outer_size;
    }
    return CSINN_TRUE;
}
