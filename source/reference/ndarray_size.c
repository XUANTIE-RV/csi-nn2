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

/* CSI-NN2 version 1.8.x */

#include "csi_ref.h"
#include "csi_utils.h"

int csi_ref_ndarray_size_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct ndarray_size_params *params)
{
    float *output_data = output->data;
    output_data[0] = csi_tensor_size(input);
    return CSINN_TRUE;
}

int csi_ref_ndarray_size_u8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct ndarray_size_params *params)
{
    uint8_t *output_data = output->data;
    output_data[0] = csi_tensor_size(input);
    return CSINN_TRUE;
}

int csi_ref_ndarray_size_i8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct ndarray_size_params *params)
{
    int8_t *output_data = output->data;
    output_data[0] = csi_tensor_size(input);
    return CSINN_TRUE;
}

int csi_ref_ndarray_size_i32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct ndarray_size_params *params)
{
    int32_t *output_data = output->data;
    output_data[0] = csi_tensor_size(input);
    return CSINN_TRUE;
}
