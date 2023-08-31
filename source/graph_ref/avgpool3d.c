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

int shl_gref_avgpool3d(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_pool_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_AVGPOOL3D, params);
    return CSINN_TRUE;
}

int shl_gref_avgpool3d_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_pool_params *params)
{
    return shl_gref_pooling3d_infer_shape(input, output, params);
}

int shl_gref_global_avgpool3d(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_GLOBAL_AVGPOOL2D, params);
    return CSINN_TRUE;
}

int shl_gref_global_avgpool3d_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_pool_params *params)
{
    int n, c, d, h, w;
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    if (input->layout == CSINN_LAYOUT_NCDHW) {
        n = 0;
        c = 1;
        d = 2;
        h = 3;
        w = 4;
    } else if (input->layout == CSINN_LAYOUT_NDHWC) {
        n = 0;
        d = 1;
        h = 2;
        w = 3;
        c = 4;
    } else {
        shl_debug_error("%s: Invalid input tensor layout!\n", __func__);
        return CSINN_UNSUPPORT_LAYOUT;
    }
    output->dim_count = 5;
    output->dim[n] = input->dim[n];
    output->dim[c] = input->dim[c];
    output->dim[d] = 1;
    output->dim[h] = 1;
    output->dim[w] = 1;

    return CSINN_TRUE;
}
