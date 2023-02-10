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

int shl_gref_strided_slice(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_strided_slice_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_STRIDED_SLICE, params);
    return CSINN_TRUE;
}

int shl_gref_strided_slice_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_strided_slice_params *params)
{
    for (int i = 0; i < params->slice_count; i++) {
        if (params->begin[i] < -input->dim[i]) params->begin[i] = -input->dim[i];
        if (params->begin[i] < 0) params->begin[i] += input->dim[i];
        if (params->begin[i] > input->dim[i]) params->begin[i] = input->dim[i];
        if (params->end[i] < -input->dim[i]) params->end[i] = -input->dim[i];
        if (params->end[i] < 0) params->end[i] += input->dim[i];
        if (params->end[i] > input->dim[i]) params->end[i] = input->dim[i];
    }

    output->dim_count = input->dim_count;
    for (int i = 0; i < output->dim_count; i++) {
        if (i < params->slice_count) {
            int slice_size = 1 + (params->end[i] - params->begin[i] - 1) / params->stride[i];
            output->dim[i] = slice_size > 0 ? slice_size : 0;
        } else {
            output->dim[i] = input->dim[i];
        }
    }
    return CSINN_TRUE;
}
