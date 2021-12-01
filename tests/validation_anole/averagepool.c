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

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"
#include "csi_ovx.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of avgpool(anole).\n");

    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    sess->base_dtype = CSINN_DTYPE_UINT8;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    // float min_value, max_value;
    int in_size = 0, out_size = 0;

    int *buffer = read_input_data_f32(argv[1]);

    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim[0] = buffer[0];          // batch
    input->dim[1] = buffer[1];          // channel
    input->dim[2] = buffer[2];          // height
    input->dim[3] = buffer[3];          // width
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];


    float *src_in   = (float *)(buffer + 14);
    uint8_t *src_tmp = malloc(in_size * sizeof(char));
    input->qinfo = get_quant_info(src_in, in_size);
    for(int i = 0; i < in_size; i++) {
        src_tmp[i] = csi_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }
    input->name = "input";


    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim[0] = input->dim[0]; // batch
    output->dim[1] = input->dim[1]; // in_channel
    output->dim[2] = buffer[12];    // out_h = (in_h + pad_top + pad_down - kernel_h) / stride_h + 1
    output->dim[3] = buffer[13];    // out_w = (in_w + pad_left + pad_right - kernel_w) / stride_w + 1
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];


    float *ref      = (float *)(buffer + 14 + in_size);
    output->qinfo = get_quant_info(ref, out_size);
    reference->data = ref;
    output->name = "output";


    struct pool_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.ceil_mode = 0;
    params.stride_height = buffer[4];
    params.stride_width  = buffer[5];
    params.filter_height = buffer[6];
    params.filter_width  = buffer[7];
    params.pad_left  = buffer[8];
    params.pad_right = buffer[9];
    params.pad_top   = buffer[10];
    params.pad_down  = buffer[11];


    if (csi_averagepool_init(input, output, &params) != CSINN_TRUE) {
        printf("avgpool init fail.\n\t");
        return -1;
    }

    csi_ovx_set_tensor(input, sess);
    csi_set_input(0, input, sess);

    csi_averagepool(input, output, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    struct csi_tensor *input_tensor = csi_alloc_tensor(NULL);
    input_tensor->data = src_tmp;
    csi_update_input(0, input_tensor, sess);
    csi_session_run(sess);

    struct csi_tensor *output_tensor = csi_alloc_tensor(NULL);
    output_tensor->is_const = 0;
    int output_num = csi_get_output_number(sess);
    printf("output_num = %d\n", output_num);
    csi_get_output(0, output_tensor, sess);

    output_tensor->qinfo->multiplier = output->qinfo->multiplier; 
    output_tensor->qinfo->shift = output->qinfo->shift; 
    output_tensor->qinfo->zero_point = output->qinfo->zero_point;
    output_tensor->dtype == CSINN_DTYPE_UINT8;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    result_verify_8(reference->data, output_tensor, input->data, difference, out_size, false);

    /* free alloced memory */
    free(buffer);
    free(input_tensor->qinfo);
    free(input_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}




