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

int shl_gref_matmul(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                    struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    shl_gref_diso_op(mat0, mat1, output, CSINN_OP_MATMUL, params);
    return CSINN_TRUE;
}

/* TODO: support onnx/numpy matmul */
int shl_gref_matmul_infer_shape(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                                struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(mat0);
    shl_tensor_try_nc1xc0_to_ndarray_shape(mat1);

    output->dim_count = mat0->dim_count > mat1->dim_count ? mat0->dim_count : mat1->dim_count;
    for (int i = 0; i < output->dim_count - 2; i++) {
        const int d1 = mat0->dim_count - 3 - i;
        const int d2 = mat1->dim_count - 3 - i;
        const int s1 = d1 >= 0 ? mat0->dim[d1] : 1;
        const int s2 = d2 >= 0 ? mat1->dim[d2] : 1;
        if (s1 == s2) {
            output->dim[output->dim_count - 3 - i] = s1;
        } else if (s1 == 1) {
            output->dim[output->dim_count - 3 - i] = s2;
        } else if (s2 == 1) {
            output->dim[output->dim_count - 3 - i] = s1;
        } else {
            shl_debug_error("%s: Invalid shapes for matmul broadcast!\n", __func__);
            return CSINN_FALSE;
        }
    }
    output->dim[output->dim_count - 2] = mat0->dim[mat0->dim_count - (params->trans_a ? 1 : 2)];
    output->dim[output->dim_count - 1] = mat1->dim[mat1->dim_count - (params->trans_b ? 2 : 1)];
    return CSINN_TRUE;
}
