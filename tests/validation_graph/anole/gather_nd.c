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

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of gather_nd(anole).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int input_dim_count = buffer[0];
    int indices_dim_count = buffer[1 + input_dim_count];

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 1, indices_size = 1, out_size = 1;

    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    csi_session_init(sess);
    csi_set_input_number(2, sess);
    csi_set_output_number(1, sess);


    /* input tensor configuration */
    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim_count = input_dim_count;
    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[1 + i];
        in_size *= input->dim[i];
    }
    input->name = "input";
    float *input_data   = (float *)(buffer + 2 + input_dim_count + indices_dim_count);
    input->data = input_data;
    get_quant_info(input);

    uint8_t *src_tmp = malloc(in_size * sizeof(char));
    for(int i = 0; i < in_size; i++) {
        src_tmp[i] = csi_ref_quantize_f32_to_u8(input_data[i], input->qinfo);
    }


    /* input tensor configuration */
    struct csi_tensor *indices  = csi_alloc_tensor(sess);
    indices->dim_count = indices_dim_count;
    for (int i = 0; i < indices->dim_count; i++) {
        indices->dim[i] = buffer[i + 2 + input->dim_count];
        indices_size *= indices->dim[i];
    }
    indices->name = "indices";
    indices->dtype = CSINN_DTYPE_INT32;
    int32_t *indices_data   = (int32_t *)(buffer + 2 + input_dim_count + indices_dim_count + in_size);
    indices->data = indices_data;


    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim_count = 0;
    int coord_dim = indices->dim[indices->dim_count - 1];
    for (int i = 0; i < indices->dim_count - 1; i++) {
        output->dim_count++;
        output->dim[i] = indices->dim[i];
    }
    for (int i = coord_dim; i < input->dim_count; i++) {
        output->dim[output->dim_count] = input->dim[i];
        output->dim_count++;
    }
    for (int i = 0; i < output->dim_count; i++) {
        out_size *= output->dim[i];
    }

    reference->data = (float *)(buffer + 2 + input->dim_count + indices->dim_count + indices_size + in_size);
    output->data = reference->data;
    output->name = "output";
    get_quant_info(output);


    /* operator parameter configuration */
    struct gather_nd_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.run_mode = CSINN_RM_NPU_GRAPH;


    if (csi_gather_nd_init(input, indices, output, &params) != CSINN_TRUE) {
        printf("gather_nd init fail.\n\t");
        return -1;
    }

    csi_set_tensor_entry(input, sess);
    csi_set_tensor_entry(indices, sess);

    csi_set_input(0, input, sess);
    csi_set_input(1, indices, sess);

    csi_gather_nd(input, indices, output, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    struct csi_tensor *input_tensor = csi_alloc_tensor(NULL);
    struct csi_tensor *indices_tensor = csi_alloc_tensor(NULL);
    input_tensor->data = src_tmp;
    indices_tensor->data = indices_data;
    csi_update_input(0, input_tensor, sess);
    csi_update_input(1, indices_tensor, sess);
    csi_session_run(sess);

    struct csi_tensor *output_tensor = csi_alloc_tensor(NULL);
    output_tensor->data = NULL;
    output_tensor->dtype = sess->base_dtype;
    output_tensor->is_const = 0;
    int output_num = csi_get_output_number(sess);
    printf("output_num = %d\n", output_num);
    csi_get_output(0, output_tensor, sess);
    memcpy(output_tensor->qinfo, output->qinfo, sizeof(struct csi_quant_info));

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    result_verify_8(reference->data, output_tensor, input->data, difference, out_size, false);

    for (int i = 0; i<output->dim_count; i++) {
        printf("%d\n",output->dim[i]);
    }

    /* free alloced memory */
    free(buffer);
    free(input_tensor->qinfo);
    free(input_tensor);
    free(indices_tensor->qinfo);
    free(indices_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);
    free(reference->qinfo);
    free(reference);
    free(src_tmp);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
