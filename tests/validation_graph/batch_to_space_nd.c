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

int main(int argc, char** argv)
{
    init_testsuite("Testing function of batch_to_space_nd(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    float min_value, max_value;
    int in_size = 1, out_size = 1;
    int prod_block = 1;
    int spatial_shape_cnt = buffer[0];
    int remain_shape_cnt  = buffer[1];
    int32_t *block_shape = (int32_t *)malloc(spatial_shape_cnt * sizeof(int32_t));
    int32_t *crops = (int32_t *)malloc(2 * spatial_shape_cnt * sizeof(int32_t));

    for(int i = 0; i < spatial_shape_cnt; i++) {
        block_shape[i] = buffer[2 + 1 + spatial_shape_cnt + remain_shape_cnt + 3 * i];
        crops[2 * i] = buffer[2 + 1 + spatial_shape_cnt + remain_shape_cnt + 3 * i + 1];
        crops[2 * i + 1] = buffer[2 + 1 + spatial_shape_cnt + remain_shape_cnt + 3 * i + 2];
        prod_block *= block_shape[i];
    }
    enum csinn_dtype_enum test_dtype = CSINN_TEST_DTYPE;
    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_LIGHT;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);


    /* input tensor configuration */
    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim_count = 1 + spatial_shape_cnt + remain_shape_cnt;    // batch_cnt + spatial_shape_cnt + remain_shape_cnt
    for(int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 2];
        in_size *= input->dim[i];
    }
    input->name = "input";
    float *input_data = (float *)(buffer + 2 + spatial_shape_cnt * 3 + input->dim_count);
    input->data = input_data;
    get_quant_info(input);
    input->dtype = CSINN_DTYPE_FLOAT32;

    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim_count = 1 + spatial_shape_cnt + remain_shape_cnt;   // output->dim_cnt = input->dim_cnt
    output->dim[0] = input->dim[0] / prod_block;      // batch_out
    output->dim[1] = input->dim[1];
    for(int i = 0; i < 2; i++) {
        output->dim[2 + i] = input->dim[2 + i] * block_shape[i] - crops[2 * i] - crops[ 2 * i + 1];
    }

    for(int i = 0; i < output->dim_count; i++) {
        out_size *= output->dim[i];
    }
    reference->data = (float *)(buffer + 2 + spatial_shape_cnt * 3 + input->dim_count + in_size);
    output->data = reference->data;
    output->name = "output";
    get_quant_info(output);


    /* operator parameter configuration */
    struct batch_to_space_nd_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.block_shape = block_shape;
    params.crops = crops;
    params.spatial_dim_cnt = spatial_shape_cnt;
    struct csi_tensor *input_tensor = convert_input(input, test_dtype);
    input->dtype = sess->base_dtype;

    if (csi_batch_to_space_nd_init(input, output, &params) != CSINN_TRUE) {
        printf("batch_to_space_nd init fail.\n\t");
        return -1;
    }


    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_batch_to_space_nd(input, output, &params);

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

    /* FIX ME */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output_tensor);
    result_verify_f32(reference->data, foutput->data, input->data, difference, out_size, false);

    /* free alloced memory */
    free(buffer);
    free(input_tensor->qinfo);
    free(input_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);
    free(reference->qinfo);
    free(reference);
    free(block_shape);
    free(crops);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
