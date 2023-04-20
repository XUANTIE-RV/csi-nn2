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

int shl_gref_space_to_batch(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_space_to_batch_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_SPACE_TO_BATCH, params);
    return CSINN_TRUE;
}

/* Only support NCHW/NHWC layout */
int shl_gref_space_to_batch_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_space_to_batch_params *params)
{
    int h, w, c;
    if (output->layout == CSINN_LAYOUT_NCHW) {
        c = 1;
        h = 2;
        w = 3;
    } else if (output->layout == CSINN_LAYOUT_NHWC) {
        h = 1;
        w = 2;
        c = 3;
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }

    int32_t block_size = params->block_size;

    output->dim_count = input->dim_count;
    output->dim[0] = input->dim[0] * block_size * block_size;
    output->dim[c] = input->dim[c];
    output->dim[h] = (input->dim[h] + params->pad_top + params->pad_bottom) / block_size;
    output->dim[w] = (input->dim[w] + params->pad_left + params->pad_right) / block_size;

    return CSINN_TRUE;
}