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
    init_testsuite("Testing function of add(anole).\n");

    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    sess->base_dtype = CSINN_DTYPE_UINT8;
    csi_session_init(sess);
    csi_set_input_number(2, sess);
    csi_set_output_number(1, sess);

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in0_size = 0, in1_size = 0, out_size = 0;

    int *buffer = read_input_data_f32(argv[1]);
    int flag  = buffer[4];

    struct csi_tensor *input0  = csi_alloc_tensor(sess);
    input0->dim[0] = buffer[0];          // batch
    input0->dim[1] = buffer[1];          // channel
    input0->dim[2] = buffer[2];          // height
    input0->dim[3] = buffer[3];          // width
    input0->dim_count = 4;
    in0_size = input0->dim[0] * input0->dim[1] * input0->dim[2] * input0->dim[3];
    input0->dtype = CSINN_DTYPE_UINT8;

    input0->name = "input0";
    float *src0_in   = (float *)(buffer + 5);
    uint8_t *src0_tmp = malloc(in0_size * sizeof(char));
    input0->qinfo = get_quant_info(src0_in, in0_size);
    for(int i = 0; i < in0_size; i++) {
        src0_tmp[i] = csi_ref_quantize_f32_to_u8(src0_in[i], input0->qinfo);
    }


    struct csi_tensor *input1  = csi_alloc_tensor(sess);
    if(flag) {
        input1->dim[0] = input0->dim[3];
        input1->dim_count = 1;
        in1_size = input1->dim[0];
    } else {
        input1->dim[0] = input0->dim[0];
        input1->dim[1] = input0->dim[1];
        input1->dim[2] = input0->dim[2];
        input1->dim[3] = input0->dim[3];
        input1->dim_count = 4;
        in1_size = in0_size;
    }

    input1->name = "input1";
    float *src1_in  = (float *)(buffer + 5 + in0_size);
    uint8_t *src1_tmp  = malloc(in1_size * sizeof(char));
    input1->qinfo = get_quant_info(src1_in, in1_size);
    for(int i = 0; i < in1_size; i++) {
        src1_tmp[i] = csi_ref_quantize_f32_to_u8(src1_in[i], input1->qinfo);
    }

    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim[0] = input0->dim[0];
    output->dim[1] = input0->dim[1];
    output->dim[2] = input0->dim[2];
    output->dim[3] = input0->dim[3];
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    float *ref      = (float *)(buffer + 5 + in0_size + in1_size);

    output->name = "output";
    output->qinfo = get_quant_info(ref, out_size);
    reference->data = ref;


    struct diso_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.run_mode = CSINN_RM_NPU_GRAPH;

    if (csi_add_init(input0, input1, output, &params) != CSINN_TRUE) {
        printf("add init fail.\n\t");
        return -1;
    }

    csi_ovx_set_tensor(input0, sess);
    csi_ovx_set_tensor(input1, sess);
    csi_set_input(0, input0, sess);
    csi_set_input(1, input1, sess);

    csi_add(input0, input1, output, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    struct csi_tensor *input0_tensor = csi_alloc_tensor(NULL);
    struct csi_tensor *input1_tensor = csi_alloc_tensor(NULL);
    input0_tensor->data = src0_tmp;
    input1_tensor->data = src1_tmp;
    csi_update_input(0, input0_tensor, sess);
    csi_update_input(1, input1_tensor, sess);
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
    result_verify_8(reference->data, output_tensor, input0->data, difference, out_size, false);


    /* free alloced memory */
    free(buffer);
    free(input0_tensor->qinfo);
    free(input0_tensor);
    free(input1_tensor->qinfo);
    free(input1_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
