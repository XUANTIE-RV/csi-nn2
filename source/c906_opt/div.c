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

/* CSI-NN2 version 2.0.x */

#include "shl_c906.h"
int shl_c906_div_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    if (input1->dtype == CSINN_DTYPE_FLOAT32) {
        float *ptr = input1->data;
        size_t tensor_size = csinn_tensor_size(input1);
        for (size_t i = 0; i < tensor_size; i++) {
            ptr[i] = 1.f / ptr[i];
        }
        cb->exec = shl_c906_mul_f32;
    } else if (input1->dtype == CSINN_DTYPE_FLOAT16) {
        __fp16 *ptr = input1->data;
        size_t tensor_size = csinn_tensor_size(input1);
        for (size_t i = 0; i < tensor_size; i++) {
            ptr[i] = 1.f / ptr[i];
        }
        cb->exec = shl_c906_mul_fp16;
    }
    return CSINN_TRUE;
}
