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

int shl_gref_space_to_depth(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_space_to_depth_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_SPACE_TO_DEPTH, params);
    return CSINN_TRUE;
}

/* Only support NCHW/NHWC layout */
int shl_gref_space_to_depth_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_space_to_depth_params *params)
{
    int h, w, c;
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    if (output->layout == CSINN_LAYOUT_NCHW) {
        c = 1;
        h = 2;
        w = 3;
    } else if (output->layout == CSINN_LAYOUT_NHWC) {
        h = 1;
        w = 2;
        c = 3;
    } else {
        shl_debug_error("%s: Invalid input tensor layout!\n", __func__);
        return CSINN_UNSUPPORT_LAYOUT;
    }

    int32_t block_size = params->block_size;

    output->dim_count = input->dim_count;
    output->dim[0] = input->dim[0];
    output->dim[c] = input->dim[c] * block_size * block_size;
    output->dim[h] = input->dim[h] / block_size;
    output->dim[w] = input->dim[w] / block_size;

    return CSINN_TRUE;
}
