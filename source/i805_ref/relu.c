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

#include "csi_ref_i805.h"


int csi_ref_i805_relu_q7(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct relu_params *params)
{
    q7_t *input_data = (q7_t *)input->data;
    int size = csi_tensor_size(input);
    csi_relu_q7(input_data, size);
    output->data = input_data;
    return CSINN_TRUE;
}

int csi_ref_i805_relu_q15(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct relu_params *params)
{
    q15_t *input_data = (q15_t *)input->data;
    int size = csi_tensor_size(input);
    csi_relu_q15(input_data, size);
    output->data = input_data;
    return CSINN_TRUE;
}
