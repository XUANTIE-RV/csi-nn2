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

#include "reference/ref.h"

int shl_ref_flatten_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    // if (input->quant_channel == output->quant_channel) {
    //     int quant_size = input->quant_channel * sizeof(struct csinn_quant_info);
    //     int t = memcmp(input->qinfo, output->qinfo, quant_size);
    //     if (t == 0) {
    //         cb->exec = shl_ref_flatten;
    //         return CSINN_TRUE;
    //     }
    // }
    cb->exec = shl_ref_flatten_quant;
    return CSINN_TRUE;
}

int shl_ref_flatten(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_flatten_params *params)
{
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;

    if (input_data == output_data) {
        return CSINN_TRUE;
    }

    int size = csinn_tensor_byte_size(input);

    memcpy(output_data, input_data, size);
    return CSINN_TRUE;
}

int shl_ref_flatten_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_flatten_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_flatten);
}
