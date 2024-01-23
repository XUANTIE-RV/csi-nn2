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

int shl_gref_embedding(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params)
{
    shl_gref_diso_op(input0, input1, output, CSINN_OP_EMBEDDING, params);
    return CSINN_TRUE;
}

int shl_gref_embedding_infer_shape(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                   struct csinn_tensor *output, struct csinn_diso_params *params)
{
    if (input0->dim_count != 1) {
        shl_debug_error("%s: unsupport dim_count = %d\n", __func__, input0->dim_count);
        return CSINN_FALSE;
    }

    output->dim_count = 2;
    output->dim[0] = input0->dim[0];
    output->dim[1] = input1->dim[1];

    SHL_DEBUG_CALL(shl_diso_debug_info(input0, input1, output, params, __func__));

    return CSINN_TRUE;
}
