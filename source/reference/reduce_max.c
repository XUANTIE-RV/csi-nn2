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

/* CSI-NN2 version 1.10.x */

#include "csi_ref.h"
#include "csi_utils.h"

int csi_ref_reduce_max_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct reduce_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    assert(params->axis_count==1);  //the Function realization assumption axis_count=1
    //axis=none
    if(*(params->axis) == -1) {
        int size = 1;
        for(int i = 0; i < input->dim_count; i++) {
            size = size * input->dim[i];
        }
        float res = *input_data;
        for(int j = 1; j < size; j++) {
            res = fmax(res, input_data[j]);
        }
        *output_data = res;
    } else {
        int axis = *(params->axis);
        int64_t outer_size = 1;
        for(int i = 0; i < axis; i++) {
            outer_size *= input->dim[i];
        }
        int64_t inner_size = 1;
        for(int i = axis + 1; i < input->dim_count; i++) {
            inner_size *= input->dim[i];
        }
        int cnt = input->dim[axis];

        for(int i = 0; i < outer_size; i++) {
            for(int k = 0; k < inner_size; k++) {
                float temp = *(input_data + k);
                for(int j = 1; j < cnt; j++) {
                    temp = fmax(temp, *(input_data + j * inner_size + k));
                }
                *(output_data + k) = temp;
            }
            input_data += inner_size * cnt;
            output_data += inner_size;
        }
    }
    return CSINN_TRUE;
}

int csi_ref_reduce_max_quant(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct reduce_params *params)
{
    return csi_ref_siso_callback_base(input, output, params, csi_ref_reduce_max_f32);
}
