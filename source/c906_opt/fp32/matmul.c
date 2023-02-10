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

#include "shl_c906.h"

/*************************************************************
  Matmul fp32 performance on C906@1GHz
  -------------------------
  |      mkn     | GFlops |
  |-----------------------|
  |   49,32,49   |  0.94  |
  |   8,176,176  |  0.85  |
  |  8,1584,176  |  0.93  |
  | 384,512,512  |  1.29  |
  | 196,1536,384 |  1.28  |
  -------------------------
 ************************************************************/

#define MATMUL_M_BLK 32
#define MATMUL_K_BLK 64
#define MATMUL_N_BLK 64

int shl_c906_matmul_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                         struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    return shl_rvv_matmul_block_fp32(mat0, mat1, output, params, MATMUL_M_BLK, MATMUL_K_BLK,
                                     MATMUL_N_BLK);
}

int shl_rvv_matmul_init_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                             struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    if (mat0->dtype == CSINN_DTYPE_FLOAT32) {
        if (mat1->dtype == CSINN_DTYPE_FLOAT32) {
            if (mat1->is_const) {
                shl_rvv_matmul_reorder_weight_fp32(mat1, MATMUL_K_BLK, MATMUL_N_BLK);
            }
            cb->exec = shl_c906_matmul_fp32;
        } else {
            shl_debug_error("mat1 unsupported dtype: %d\n", mat1->dtype);
            return CSINN_FALSE;
        }
    } else {
        shl_debug_error("mat0 unsupported dtype: %d\n", mat0->dtype);
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}
