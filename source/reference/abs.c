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

int csi_ref_abs_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    int size = csi_tensor_size(input);

    for (int i = 0; i < size; i++) {
        output_data[i] = fabs(input_data[i]);
    }
    return CSINN_TRUE;
}

int csi_ref_abs_quant(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params)
{
    return csi_ref_siso_callback_base(input, output, params, csi_ref_abs_f32);
}
