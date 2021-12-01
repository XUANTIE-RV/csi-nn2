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

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"
#include "../valid_data/clip_u8.dat"

static void verify_clip_u8(float *input_data,
                           float *ref_data,
                           float clip_fmin,
                           float clip_fmax,
                           int32_t size,
                           float difference)
{
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size, out_size;

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = 1;
    input->dim[2] = 1;
    input->dim[3] = size;
    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_UINT8;
    input->layout = CSINN_LAYOUT_NHWC;
    input->name = "input";
    input->data = (float *)input_data;
    get_quant_info(input);
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];

    uint8_t *src_tmp = malloc(in_size * sizeof(char));
    for(int i = 0; i < in_size; i++) {
        src_tmp[i] = csi_ref_quantize_f32_to_u8(input_data[i], input->qinfo);
    }
    input->data = src_tmp;

    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = 1;
    output->dim[2] = 1;
    output->dim[3] = size;
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NHWC;
    output->name = "output";
    output->data = (float *)ref_data;
    get_quant_info(output);
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    output->data = malloc(out_size);

    struct clip_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NHWC;
    params.base.run_mode = CSINN_RM_LAYER;
    params.max_value = clip_fmax;
    params.min_value = clip_fmin;

    if (csi_clip_init(input, output, &params) == CSINN_TRUE) {
        csi_clip(input, output, &params);
    }

    reference->data  = (float *)ref_data;
    result_verify_8(reference->data, output, input->data, difference, out_size, false);
    free(input);
    free(output->data);
    free(output);
    free(reference);
    free(src_tmp);
}


int main(int argc, char** argv)
{
    init_testsuite("Testing function of relu(u8) for i805.\n");
    verify_clip_u8(clip_input_0, clip_output_0, 0.0, 6.0, 79, 1.0);
}
