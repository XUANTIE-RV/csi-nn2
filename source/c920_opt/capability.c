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

#include "c920/cap.h"

static int common_all_support(struct csinn_tensor *input, struct csinn_params_base *base)
{
    if ((input->dtype != CSINN_DTYPE_FLOAT16) && (input->dtype != CSINN_DTYPE_FLOAT32)) {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_ASM;
}

int shl_c920_conv2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv2d_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c920_matmul_cap(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    int batches_a = 1;
    int batches_b = 1;

    /* compute the outer size */
    for (int i = 0; i < mat0->dim_count - 2; i++) {
        batches_a *= mat0->dim[i];
    }
    for (int i = 0; i < mat1->dim_count - 2; i++) {
        batches_b *= mat1->dim[i];
    }

    if (mat0->dtype == CSINN_DTYPE_FLOAT32 && mat1->dtype == CSINN_DTYPE_FLOAT32 ||
        mat0->dtype == CSINN_DTYPE_FLOAT16 &&
            (mat1->dtype == CSINN_DTYPE_FLOAT16 || mat1->dtype == CSINN_DTYPE_INT8)) {
        if (!params->trans_a && !params->trans_b) {
            if (batches_a == batches_b) {
                return CSINN_OPT_INTRINSIC;
            } else if (batches_a > 1 && batches_b == 1) {
                return CSINN_OPT_INTRINSIC;
            }
        }
    }

    return CSINN_OPT_UNSUPPORTED;
}
