/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

#include "csi_ref.h"
#include <assert.h>

int csi_ref_not_u32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params)

{
    uint32_t *input_data = input->data;
    uint32_t *output_data = output->data;
    int size = csi_tensor_size(input);

    for (int i = 0; i < size; i++) {
        output_data[i] = ~(input_data[i]);
    }
    return CSINN_TRUE;
}

int csi_ref_not_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params)
{
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    int size = csi_tensor_size(input);

    for (int i = 0; i < size; i++) {
        output_data[i] = ~(input_data[i]);
    }
    return CSINN_TRUE;
}

int csi_ref_not_i8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params)
{
    int8_t *input_data = input->data;
    int8_t *output_data = output->data;
    int size = csi_tensor_size(input);

    for (int i = 0; i < size; i++) {
        output_data[i] = ~(input_data[i]);
    }
    return CSINN_TRUE;
}
