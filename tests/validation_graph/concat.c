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
    init_testsuite("Testing function of concat(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int input_cnt = buffer[4];
    int axis = buffer[5];

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 0, out_size = 1;

    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_API;
    csi_session_init(sess);
    csi_set_input_number(input_cnt, sess);
    csi_set_output_number(1, sess);

    /* input tensor configuration */
    struct csi_tensor *input[input_cnt];
    float *input_data[input_cnt];
    void **src_tmp = malloc(input_cnt * sizeof(void *));
    char input_name[input_cnt][10];
    for(int i = 0; i < input_cnt; i++) {

        input[i]  = csi_alloc_tensor(sess);
        input[i]->dim[0] = buffer[0];          // batch
        input[i]->dim[1] = buffer[1];          // in_channel
        input[i]->dim[2] = buffer[2];          // height
        input[i]->dim[3] = buffer[3];          // width
        input[i]->dim_count = 4;
        in_size = input[i]->dim[0] * input[i]->dim[1] * input[i]->dim[2] * input[i]->dim[3];

        input_data[i] = (float *)(buffer + 6 + in_size * i);
        input[i]->data = input_data[i];
        get_quant_info(input[i]);
        sprintf(input_name[i], "input_%d", i);
        input[i]->name = input_name[i];

        src_tmp[i] = malloc(in_size * sizeof(char));
        for(int j = 0; j < in_size; j++) {
            if (sess->base_dtype == CSINN_DTYPE_UINT8) {
                *((uint8_t *)src_tmp[i] + j) = csi_ref_quantize_f32_to_u8(input_data[i][j], input[i]->qinfo);
            } else if (sess->base_dtype == CSINN_DTYPE_INT8) {
                *((int8_t *)src_tmp[i] + j) = csi_ref_quantize_f32_to_i8(input_data[i][j], input[i]->qinfo);
            }
        }
    }

    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(sess);
    for(int i = 0; i < 4; i++) {
        if(i == axis) {
            output->dim[i] = input_cnt * buffer[i];
        } else {
            output->dim[i] = buffer[i];
        }
        out_size *= output->dim[i];
    }
    output->dim_count = 4;

    reference->data = (float *)(buffer + 6 + in_size * input_cnt);
    output->data = reference->data;
    get_quant_info(output);
    output->name = "output";


    /* operator parameter configuration */
    struct concat_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.axis = axis;
    params.inputs_count = input_cnt;


    if (csi_concat_init((struct csi_tensor **)&input, output, &params) != CSINN_TRUE) {
        printf("concat init fail.\n\t");
        return -1;
    }

    for(int i = 0; i < input_cnt; i++) {
        csi_set_tensor_entry(input[i], sess);
        csi_set_input(i, input[i], sess);
    }

    csi_concat((struct csi_tensor **)&input, output, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    struct csi_tensor *input_tensor[input_cnt];
    for (int i = 0; i < input_cnt; i++) {
        input_tensor[i] = csi_alloc_tensor(NULL);
    }
    for(int i = 0; i < input_cnt; i++) {
        if (sess->base_dtype == CSINN_DTYPE_FLOAT32) {
            input_tensor[i]->data = input_data[i];
        } else if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
            input_tensor[i]->data = src_tmp[i];
        }
        csi_update_input(i, input_tensor[i], sess);
    }
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
        result_verify_8(reference->data, output_tensor, input[0]->data, difference, out_size, false);
    } else if (sess->base_dtype == CSINN_DTYPE_FLOAT32) {
        result_verify_f32(reference->data, output_tensor->data, input[0]->data, difference, out_size, false);
    }

    /* free alloced memory */
    free(buffer);
    for (int i = 0; i < input_cnt; i++) {
        free(input_tensor[i]->qinfo);
        free(input_tensor[i]);
    }
    free(output_tensor->qinfo);
    free(output_tensor);
    free(reference->qinfo);
    free(reference);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
