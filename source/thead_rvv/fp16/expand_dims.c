/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "rvv/rvv.h"

int shl_rvv_expand_dims_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_expand_dims_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    int size = 1;
    if (input_data != output_data) {
        for (int i = 0; i < input->dim_count; i++) {
            size *= input->dim[i];
        }
        int j = 0;
        while (j < size) {
            int vl = vsetvl_e16m4(size - j);
            vfloat16m4_t _in = vle16_v_f16m4(input_data, vl);
            vse16_v_f16m4(output_data, _in, vl);
            input_data += vl;
            output_data += vl;
            j += vl;
        }
    }
    return CSINN_TRUE;
}
