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

int shl_gref_sequence_mask(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_sequence_mask_params *params)
{
    shl_gref_diso_op(input0, input1, output, CSINN_OP_SEQUENCE_MASK, params);
    return CSINN_TRUE;
}

int shl_gref_sequence_mask_infer_shape(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                       struct csinn_tensor *output,
                                       struct csinn_sequence_mask_params *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(input0);
    shl_tensor_try_nc1xc0_to_ndarray_shape(input1);
    int maxlen = 0;
    if (input1->dim_count > 0) {
        int32_t *input1_data = (int32_t *)input1->data;
        maxlen = input1_data[0];
    } else {
        int in_size = 1;
        for (int i = 0; i < input0->dim_count; i++) {
            in_size *= input0->dim[i];
        }
        int32_t *lengths = (int32_t *)input0->data;
        for (int i = 0; i < in_size; i++) {
            maxlen = lengths[i] > maxlen ? lengths[i] : maxlen;
        }
    }

    output->dim_count = input1->dim_count + 1;
    if (params->axis == -1) {
        for (int i = 0; i < input1->dim_count; i++) {
            output->dim[i] = input1->dim[i];
        }
        output->dim[output->dim_count - 1] = maxlen;
    } else {
        for (int i = 0; i < output->dim_count; i++) {
            if (i < params->axis) {
                output->dim[i] = input1->dim[i];
            } else if (i == params->axis) {
                output->dim[i] = maxlen;
            } else {
                output->dim[i] = input1->dim[i - 1];
            }
        }
    }

    return CSINN_FALSE;
}
