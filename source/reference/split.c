/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#include "csi_nn.h"
#include "csi_utils.h"

int csi_split_u8(struct csi_tensor *input,
                 struct csi_tensor **output,
                 struct split_params *params)
{
    const int32_t batches = input->dim[0];
    const int32_t input_depth = input->dim[1];
    const int32_t input_height = input->dim[2];
    const int32_t input_width = input->dim[3];

    int32_t begin[4] = {0, 0, 0, 0};
    for(int i = 0; i < params->output_num; i++){
        if(i != 0){
            begin[1] = params->split_index[i-1];
        }
        int32_t end_1;
        if(i == params->output_num -1){
            end_1 = input_depth;
        }else{
            end_1 = params->split_index[i];
        }
        int32_t end[4] = {batches, end_1, input_width, input_height};
        int32_t strides[4] = {1, 1, 1, 1};
        struct csi_tensor *output_ptr = output[i];
        struct slice_params sparams;
        sparams.layout = CSINN_NCHW;
        sparams.begin = begin;
        sparams.end = end;
        sparams.strides = strides;
        sparams.api = CSINN_REF;
        csi_slice_init(input, output_ptr, &sparams);
        csi_slice(input, output_ptr, &sparams);
    }
    return CSINN_TRUE;
}

int csi_split_init(struct csi_tensor *input,
                   struct csi_tensor **output,
                   struct split_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_SPLIT, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_split(struct csi_tensor *input,
              struct csi_tensor **output,
              struct split_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
