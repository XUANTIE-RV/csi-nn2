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

/*************************************************************
    note: support flexible vlen
*************************************************************/
int shl_rvv_reshape_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    shl_gref_reshape_infer_shape(input, output, params);
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        const int packn = csrr_vlenb() / sizeof(float);  // = input->dim[input->dim_count - 1]
        const int vl = vsetvl_e32m1(packn);
        int outer_size = input->dim[0] * input->dim[1];  // batch fuse to outer
        int inner_size = 1;
        for (int i = 2; i < input->dim_count - 1; i++) {
            inner_size *= input->dim[i];
        }
        for (int bc = 0; bc < outer_size; bc++) {
            float *out_ptr = output_data + bc * inner_size * packn;
            for (int i = 0; i < inner_size; i++) {
                vfloat32m1_t _input = vle32_v_f32m1(input_data, vl);
                input_data += vl;
                vsse32_v_f32m1(out_ptr, inner_size * sizeof(float), _input, vl);
                out_ptr += 1;
            }
        }
        if (output->dim_count == 1)
            output->layout = CSINN_LAYOUT_N;
        else if (output->dim_count == 2)
            output->layout = CSINN_LAYOUT_NC;
        else if (output->dim_count == 3)
            output->layout = CSINN_LAYOUT_NCW;
        else if (output->dim_count == 4)
            output->layout = CSINN_LAYOUT_NCHW;
        else if (output->dim_count == 5)
            output->layout = CSINN_LAYOUT_NCDHW;
    } else if (input->layout >= CSINN_LAYOUT_N && input->layout <= CSINN_LAYOUT_NCDHW) {
        memcpy(output_data, input_data, csinn_tensor_byte_size(input));
        if (output->dim_count == 1)
            output->layout = CSINN_LAYOUT_N;
        else if (output->dim_count == 2)
            output->layout = CSINN_LAYOUT_NC;
        else if (output->dim_count == 3)
            output->layout = CSINN_LAYOUT_NCW;
        else if (output->dim_count == 4)
            output->layout = CSINN_LAYOUT_NCHW;
        else if (output->dim_count == 5)
            output->layout = CSINN_LAYOUT_NCDHW;
        else if (output->dim_count == 6)
            output->layout = CSINN_LAYOUT_NLCDHW;
    } else if (input->layout >= CSINN_LAYOUT_NWC && input->layout <= CSINN_LAYOUT_NDHWC) {
        memcpy(output_data, input_data, csinn_tensor_byte_size(input));
        if (output->dim_count == 1)
            output->layout = CSINN_LAYOUT_N;
        else if (output->dim_count == 2)
            output->layout = CSINN_LAYOUT_NC;
        else if (output->dim_count == 3)
            output->layout = CSINN_LAYOUT_NWC;
        else if (output->dim_count == 4)
            output->layout = CSINN_LAYOUT_NHWC;
        else if (output->dim_count == 5)
            output->layout = CSINN_LAYOUT_NDHWC;
    } else if (input->layout == CSINN_LAYOUT_NLCDHW) {
        memcpy(output_data, input_data, csinn_tensor_byte_size(input));
        if (output->dim_count == 1)
            output->layout = CSINN_LAYOUT_N;
        else if (output->dim_count == 2)
            output->layout = CSINN_LAYOUT_NC;
        else if (output->dim_count == 3)
            output->layout = CSINN_LAYOUT_NCW;
        else if (output->dim_count == 4)
            output->layout = CSINN_LAYOUT_NCHW;
        else if (output->dim_count == 5)
            output->layout = CSINN_LAYOUT_NCDHW;
        else if (output->dim_count == 6)
            output->layout = CSINN_LAYOUT_NLCDHW;
    }
    return CSINN_TRUE;
}
