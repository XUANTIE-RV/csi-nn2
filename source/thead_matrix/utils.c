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

#include "rvm/rvm.h"

int csrr_xrlenb()
{
    int a = 0;
    asm volatile("csrr %0, xrlenb" : "=r"(a) : : "memory");
    return a;
}

bool shl_rvm_get_binary_model_op_init(struct csinn_session *sess)
{
    struct shl_rvm_option *option = shl_rvm_get_graph_option(sess);
    if (option && option->binary_model_op_init) {
        return true;
    } else {
        return false;
    }
}

void shl_rvm_set_binary_model_op_init(struct csinn_session *sess, bool value)
{
    struct shl_rvm_option *option = shl_rvm_get_graph_option(sess);
    option->binary_model_op_init = value;
}