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
    init_testsuite("Testing function of crop(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int in_out_dim = buffer[0];
    int *begin = (int *)malloc(in_out_dim * sizeof(int));
    int *end = (int *)malloc(in_out_dim * sizeof(int));
    for(int i = 0; i < in_out_dim; i++) {
        begin[i] = buffer[2 + in_out_dim + 3 * i];
        end[i] = buffer[2 + in_out_dim + 3 * i + 1];
    }

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 1, out_size = 1;
    enum csinn_dtype_enum test_dtype = CSINN_TEST_DTYPE;
    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_API;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);


    /* input tensor configuration */
    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim_count = in_out_dim;
    for(int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[1 + i];
        in_size *= input->dim[i];
    }
    input->name = "input";
    float *input_data = (float *)(buffer + 3 + 4 * input->dim_count);
    input->data = input_data;
    get_quant_info(input);
    input->dtype = CSINN_DTYPE_FLOAT32;

    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim_count = in_out_dim;
    for(int i = 0; i < output->dim_count; i++) {
        output->dim[i] = end[i] - begin[i];   // end[i] - begin[i] ( stride[i] = 1 )
        out_size *= output->dim[i];
    }
    // out_size = buffer[2 + 4 * input->dim_count];
    reference->data = (float *)(buffer + 3 + 4 * input->dim_count + in_size);
    output->data = reference->data;
    output->name = "output";
    get_quant_info(output);


    /* operator parameter configuration */
    struct crop_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.axis = buffer[1 + input->dim_count];
    params.offset_num = input->dim_count - params.axis;

    int32_t *offset = (int32_t *)malloc((params.offset_num) * sizeof(int32_t));
    for(int i = 0; i < params.offset_num; i++) {
        offset[i] = begin[i + params.axis];
    }
    params.offset = offset;


    struct csi_tensor *input_tensor = convert_input(input, test_dtype);
    input->dtype = sess->base_dtype;
    /*
    light:
        1. cropping on the batch axis is not supported. -->> axis >= 1
        2. input->dim_count <= 4
    */
    if (csi_crop_init(input, output, &params) != CSINN_TRUE) {
        printf("crop init fail.\n\t");
        return -1;
    }

    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_crop(input, output, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    csi_update_input(0, input_tensor, sess);
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
    if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
        result_verify_8(reference->data, output_tensor, input->data, difference, out_size, false);
    } else if (sess->base_dtype == CSINN_DTYPE_FLOAT32 && output_tensor->dtype == CSINN_DTYPE_INT8) {
        struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output_tensor);
        result_verify_f32(reference->data, foutput->data, input->data, difference, out_size, false);
    }

    /* free alloced memory */
    free(buffer);
    free(input_tensor->qinfo);
    free(input_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);
    free(reference->qinfo);
    free(reference);
    free(begin);
    free(end);
    free(offset);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
