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

/* CSI-NN2 version 1.12.x */

#include "csi_ref.h"

int csi_ref_asinh_f32(struct csi_tensor *input, struct csi_tensor *output,
                      struct siso_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int size = csi_tensor_size(input);

    for (int i = 0; i < size; i++) {
        output_data[i] = asinh(input_data[i]);
    }
    return CSINN_TRUE;
}

int csi_ref_asinh_quant(struct csi_tensor *input, struct csi_tensor *output,
                        struct siso_params *params)
{
    return csi_ref_siso_callback_base(input, output, params, csi_ref_asinh_f32);
}
