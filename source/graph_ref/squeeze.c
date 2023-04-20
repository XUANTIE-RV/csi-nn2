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

int shl_gref_squeeze(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_squeeze_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_SQUEEZE, params);
    return CSINN_TRUE;
}

int shl_gref_squeeze_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_squeeze_params *params)
{
    if (params->axis_num == -1) {
        int j = 0;
        for (int i = 0; i < input->dim_count; i++) {
            if (input->dim[i] != 1) {
                output->dim[j++] = input->dim[i];
            }
        }
        output->dim_count = j;
    } else {
        output->dim_count = input->dim_count - params->axis_num;
        int j = 0;
        int k = 0;
        for (int i = 0; i < input->dim_count; i++) {
            if (i == params->axis[k]) {
                k += 1;
            } else {
                output->dim[j++] = input->dim[i];
            }
        }
    }
    return CSINN_TRUE;
}
