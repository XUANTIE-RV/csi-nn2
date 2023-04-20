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
    output->dim_count = mat0->dim_count;
    for (int i = 0; i < output->dim_count - 2; i++) {
        output->dim[i] = mat0->dim[i];
    }
    output->dim[output->dim_count - 2] = mat0->dim[mat0->dim_count - (params->trans_a ? 1 : 2)];
    output->dim[output->dim_count - 1] = mat1->dim[mat1->dim_count - (params->trans_b ? 2 : 1)];
    return CSINN_TRUE;
}
