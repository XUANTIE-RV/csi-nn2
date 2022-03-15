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
    init_testsuite("Testing function of unstack(anole).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int axis = buffer[0];
    int input_dims = buffer[1];
    int output_cnt = buffer[2 + axis];

    struct csi_tensor *reference[output_cnt];
    for(int i = 0; i < output_cnt; i++) {
        reference[i] = csi_alloc_tensor(NULL);
    }
    int in_size = 1, out_size = 1;

    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(output_cnt, sess);

    /* input tensor configuration */
    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim_count = input_dims;
    for(int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[2 + i];
        in_size *= input->dim[i];
    }
    input->name = "input";
    float *input_data = (float *)(buffer + 2 + input_dims);
    input->data = input_data;
    get_quant_info(input);

    uint8_t *src_tmp = malloc(in_size * sizeof(uint8_t));
    for (int i = 0; i < in_size; i++) {
        src_tmp[i] = csi_ref_quantize_f32_to_u8(input_data[i], input->qinfo);
    }


    /* output tensor configuration */
    struct csi_tensor *output[output_cnt];
    char output_name[output_cnt][10];
    out_size = in_size / output_cnt;
    for(int i = 0; i < output_cnt; i++) {
        output[i] = csi_alloc_tensor(sess);
        output[i]->dim_count = input_dims - 1;
        for(int j = 0; j < input_dims; j++) {
            if(j < axis) {
                output[i]->dim[j] = input->dim[j];
            } else if(j > axis) {
                output[i]->dim[j-1] = input->dim[j];
            }
        }
        reference[i]->data = (float *)(buffer + 2 + input_dims + in_size +  out_size * i);
        output[i]->data = reference[i]->data;
        get_quant_info(output[i]);
        sprintf(output_name[i], "output_%d", i);
        output[i]->name = output_name[i];
    }


    /* operator parameter configuration */
    struct unstack_params params;
    params.base.api = CSINN_API;
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.base.name = "params";
    params.axis = axis;
    params.outputs_count = output_cnt;


    if (csi_unstack_init(input, (struct csi_tensor **)&output, &params) != CSINN_TRUE) {
        printf("unstack init fail.\n\t");
        return -1;
    }

    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_unstack(input, (struct csi_tensor **)&output, &params);

    for(int i = 0; i < output_cnt; i++) {
        csi_set_output(i, output[i], sess);
    }
    csi_session_setup(sess);

    struct csi_tensor *input_tensor = csi_alloc_tensor(NULL);
    input_tensor->data = src_tmp;
    csi_update_input(0, input_tensor, sess);
    csi_session_run(sess);



    struct csi_tensor *output_tensor[output_cnt];
    for(int i = 0; i < output_cnt; i++) {
        output_tensor[i] = csi_alloc_tensor(NULL);
        output_tensor[i]->data = NULL;
        output_tensor[i]->dtype = sess->base_dtype;
        output_tensor[i]->is_const = 0;
        memcpy(output_tensor[i]->qinfo, output[i]->qinfo, sizeof(struct csi_quant_info));
    }
    int output_num = csi_get_output_number(sess);
    printf("output_num = %d\n", output_num);
    for(int i = 0; i < output_num; i++) {   // output_cnt = output_num
        csi_get_output(i, output_tensor[i], sess);
    }


    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    for(int i = 0; i < output_cnt; i++) {
        result_verify_8(reference[i]->data, output_tensor[i], input->data, difference, out_size, false);
    }

    /* free alloced memory */
    free(buffer);
    free(input_tensor->qinfo);
    free(input_tensor);
    for(int i = 0; i < output_cnt; i++) {
        free(output_tensor[i]->qinfo);
        free(output_tensor[i]);
        free(reference[i]->qinfo);
        free(reference[i]);
    }
    free(src_tmp);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
