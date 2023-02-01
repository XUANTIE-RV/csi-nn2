/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#include "shl_c910.h"

#define C910_OP_PATTERN_MAX 40
static struct shl_cb_table shl_c910_cb_table[C910_OP_PATTERN_MAX];

void shl_c910_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init,
                     void *exec, void *est)
{
    static int i = 0;
    shl_c910_cb_table[i].shl_cb_key = op_name * CSINN_DTYPE_SIZE + dtype;
    shl_c910_cb_table[i].shl_cb_value.init = init;
    shl_c910_cb_table[i].shl_cb_value.exec = exec;
    shl_c910_cb_table[i].shl_cb_value.est = est;
    i++;
}

struct csinn_callback *shl_cb_map_rvv(int op, int dtype);
struct csinn_callback *shl_cb_map_c910(int op, int dtype)
{
    struct csinn_callback *cb = NULL;
    for (int i = 0; i < C910_OP_PATTERN_MAX; i++) {
        if (shl_c910_cb_table[i].shl_cb_key == (op * CSINN_DTYPE_SIZE + dtype)) {
            cb = &(shl_c910_cb_table[i].shl_cb_value);
            break;
        }
    }
    if ((cb == NULL) || (cb->est == NULL && (cb->init == NULL || cb->exec == NULL))) {
        cb = shl_cb_map_rvv(op, dtype);
    }
    return cb;
}

void shl_target_init_c910()
{
    shl_c910_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_c910_conv2d_init_fp32, NULL,
                    shl_gref_conv2d);
    shl_c910_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_c910_conv2d_init_fp16, NULL,
                    shl_gref_conv2d);

    shl_register_op_callback(CSINN_C910, shl_cb_map_c910);
    shl_register_runtime_callback(CSINN_C910, shl_gref_runtime_callback);
}
