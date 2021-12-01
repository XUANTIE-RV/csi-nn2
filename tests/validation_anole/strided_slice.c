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
    init_testsuite("Testing function of strided_slice(anole).\n");

    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    sess->base_dtype = CSINN_DTYPE_UINT8;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct strided_slice_params params;
    int in_size = 1;
    int out_size = 1;
    int zp, quantized_multiplier, shift;
    float scale, min_value, max_value;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim_count = buffer[0];
    for(int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i+1];
        in_size *= input->dim[i];
    }
    params.slice_count = buffer[1+input->dim_count];
    params.begin = (int *)malloc(params.slice_count * sizeof(int));
    params.end = (int *)malloc(params.slice_count * sizeof(int));
    params.stride = (int *)malloc(params.slice_count * sizeof(int));
    for(int i = 0; i < params.slice_count; i++) {
        params.begin[i] = buffer[2+input->dim_count+3*i];
        params.end[i] = buffer[3+input->dim_count+3*i];
        params.stride[i] = buffer[4+input->dim_count+3*i];
    }
    output->dim_count = input->dim_count;
    for(int i = 0; i < output->dim_count; i++) {
        if(i < params.slice_count) {
            output->dim[i] = (params.end[i] - params.begin[i]) / params.stride[i];
        } else {
            output->dim[i] = input->dim[i];
        }
    }
    out_size = buffer[2+input->dim_count+3*params.slice_count];
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    input->dtype = CSINN_DTYPE_UINT8;
    output->dtype = CSINN_DTYPE_UINT8;


    float *src_in   = (float *)(buffer + 3 + input->dim_count + 3*params.slice_count);
    float *ref      = (float *)(buffer + 3 + input->dim_count + 3*params.slice_count + in_size); //input->data + in_size
    uint8_t *src_tmp = malloc(in_size * sizeof(char));

    input->qinfo = get_quant_info(src_in, in_size);

    for(int i = 0; i < in_size; i++) {
        src_tmp[i] = csi_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }
    input->name = "input";

    output->qinfo = get_quant_info(ref, out_size);
    reference->data = ref;
    output->name = "output";

    if (csi_strided_slice_init(input, output, &params) != CSINN_TRUE) {
        printf("strided slice init fail.\n\t");
        return -1;
    }

    csi_ovx_set_tensor(input, sess);
    csi_set_input(0, input, sess);

    csi_strided_slice(input, output, &params);

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
    csi_get_output(0, output_tensor, sess);    // output_num = 1


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
