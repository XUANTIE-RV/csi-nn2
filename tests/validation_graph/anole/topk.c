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

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of topk(anole).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int topk = buffer[0];
    int in_out_dim = buffer[1];

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct csi_tensor *reference1 = csi_alloc_tensor(NULL);
    int in_size = 1, out_size = 1;

    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(2, sess);



    /* input tensor configuration */
    struct csi_tensor *input = csi_alloc_tensor(sess);
    input->dim_count = in_out_dim;
    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[2 + i];
        in_size *= input->dim[i];
    }
    input->name = "input";
    float *input_data   = (float *)(buffer + 2 + in_out_dim);
    input->data = input_data;
    get_quant_info(input);

    uint8_t *src_tmp = malloc(in_size * sizeof(char));
    for(int i = 0; i < in_size; i++) {
        src_tmp[i] = csi_ref_quantize_f32_to_u8(input_data[i], input->qinfo);
    }

    /* output(value) tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim_count = in_out_dim;
    for (int i = 0; i < output->dim_count - 1; i++) {
        output->dim[i] = input->dim[i];
    }
    output->dim[output->dim_count - 1] = topk;
    out_size = in_size / input->dim[input->dim_count - 1] * topk;

    reference->data = (float *)(buffer + 2 + input->dim_count + in_size);
    output->data = reference->data;
    output->name = "output_value";
    get_quant_info(output);


    /* output(index) tensor configuration */
    struct csi_tensor *output1 = csi_alloc_tensor(sess);
    output1->dim_count = in_out_dim;
    for (int i = 0; i < output1->dim_count - 1; i++) {
        output1->dim[i] = input->dim[i];
    }
    output1->dim[output1->dim_count - 1] = topk;
    reference1->data = (float *)(buffer + 2 + input->dim_count + in_size + out_size);
    output1->data = reference1->data;
    output1->name = "output_index";


    /* operator parameter configuration */
    struct topk_params params;
    params.k = topk;
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.base.api = CSINN_API;


    /*
    anole:
        only support 1D or 2D tensor for input (input->dim_count <= 2) ?
    */
    if (csi_topk_init(input, output, output1, &params) != CSINN_TRUE) {
        printf("topk init fail.\n\t");
        return -1;
    }

    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_topk(input, output, output1, &params);

    csi_set_output(0, output, sess);
    csi_set_output(1, output1, sess);
    csi_session_setup(sess);


    struct csi_tensor *input_tensor = csi_alloc_tensor(NULL);
    input_tensor->data = src_tmp;
    csi_update_input(0, input_tensor, sess);
    csi_session_run(sess);

    struct csi_tensor *output_tensor = csi_alloc_tensor(NULL);
    output_tensor->data = NULL;
    output_tensor->dtype = sess->base_dtype;
    output_tensor->is_const = 0;

    struct csi_tensor *output_tensor1 = csi_alloc_tensor(NULL);
    output_tensor1->data = NULL;
    output_tensor1->dtype = sess->base_dtype;
    output_tensor1->is_const = 0;

    int output_num = csi_get_output_number(sess);
    printf("output_num = %d\n", output_num);
    csi_get_output(0, output_tensor, sess);
    csi_get_output(1, output_tensor1, sess);
    memcpy(output_tensor->qinfo, output->qinfo, sizeof(struct csi_quant_info));

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    result_verify_8(reference->data, output_tensor, input->data, difference, out_size, false);
    /* FIXME: how to evaluate the topk value's index */
    result_verify_int32(reference1->data, output_tensor1->data, input->data, difference, out_size, false);

    /* free alloced memory */
    free(buffer);
    free(input_tensor->qinfo);
    free(input_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);
    free(output_tensor1->qinfo);
    free(output_tensor1);
    free(reference->qinfo);
    free(reference);
    free(src_tmp);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
