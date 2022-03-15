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

bool find_axis(int *axis, int axis_cnt, int index)
{
    for(int i = 0; i < axis_cnt; i++) {
        if(axis[i] == index) {
            return true;
        }
    }
    return false;
}


int main(int argc, char** argv)
{
    init_testsuite("Testing function of min(anole).\n");


    int *buffer = read_input_data_f32(argv[1]);
    bool keep_dim = buffer[4];
    int axis_count = buffer[5];
    int *axis = (int *)malloc(axis_count * sizeof(int));
    for(int i = 0; i < axis_count; i++) {
        axis[i] = buffer[6 + i];
    }

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 0, out_size = 1;

    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);


    /* input tensor configuration */
    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim[0] = buffer[0];          // batch
    input->dim[1] = buffer[1];          // channel
    input->dim[2] = buffer[2];          // height
    input->dim[3] = buffer[3];          // width
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    input->name = "input";
    float *input_data   = (float *)(buffer + 6 + axis_count);
    input->data = input_data;
    get_quant_info(input);

    uint8_t *src_tmp = malloc(in_size * sizeof(char));
    for(int i = 0; i < in_size; i++) {
        src_tmp[i] = csi_ref_quantize_f32_to_u8(input_data[i], input->qinfo);
    }


    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim[2] = input->dim[2];
    output->dim[3] = input->dim[3];
    if(keep_dim) {
        output->dim_count = input->dim_count;
        output->dim[0] = input->dim[0];     // can not reduce on batch and channel axis
        output->dim[1] = input->dim[1];
        for(int i = 2; i < output->dim_count; i++) {
            if(find_axis(axis, axis_count, i) == true) {
                output->dim[i] = 1;
            } else {
                output->dim[i] = input->dim[i];
            }
        }
    } else {
        output->dim_count = input->dim_count - axis_count;
        output->dim[0] = input->dim[0];     // can not reduce on batch and channel axis
        output->dim[1] = input->dim[1];
        int j = 2;
        for(int i = 2; i < input->dim_count; i++) {
            if(find_axis(axis, axis_count, i) == false) {
                output->dim[j] = input->dim[i];
                j++;
            }
        }
    }
    for(int i = 0; i < output->dim_count; i++) {
        out_size *= output->dim[i];
    }
    reference->data = (float *)(buffer + 6 + axis_count + in_size);
    output->data = reference->data;
    output->name = "output";
    get_quant_info(output);


    /* operator parameter configuration */
    struct reduce_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.axis = axis;
    params.axis_count = axis_count;
    params.keepdims = keep_dim;


    if (csi_min_init(input, output, &params) != CSINN_TRUE) {
        printf("min init fail.\n\t");
        return -1;
    }

    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_min(input, output, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    struct csi_tensor *input_tensor = csi_alloc_tensor(NULL);
    input_tensor->data = src_tmp;
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
    result_verify_8(reference->data, output_tensor, input->data, difference, out_size, false);

    /* free alloced memory */
    free(buffer);
    free(input_tensor->qinfo);
    free(input_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);
    free(reference->qinfo);
    free(reference);
    free(src_tmp);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
