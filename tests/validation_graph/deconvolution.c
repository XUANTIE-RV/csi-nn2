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
#include "csi_pnna.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of deconv2d(graph).\n");

    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_LIGHT;
    sess->base_dtype = CSINN_DTYPE_INT8;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    float min_value, max_value;
    int in_size = 0, out_size = 0, weight_size = 0;

    int *buffer = read_input_data_f32(argv[1]);

    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim[0]   = buffer[0];          // batch
    input->dim[1]   = buffer[1];          // in_channel
    input->dim[2]   = buffer[2];          // height
    input->dim[3]   = buffer[3];          // width
    input->dim_count = 4;
    in_size  = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];

    float *input_data = (float *)(buffer + 17);
    /* get input min max */
    find_min_max((float *)input_data, &max_value, &min_value, in_size);
    input->qinfo->min = min_value;
    input->qinfo->max = max_value;
    input->name = "input";


    struct csi_tensor *kernel  = csi_alloc_tensor(sess);
    kernel->data = (float *)(buffer + 17 + in_size);
    kernel->dim[0]  = buffer[1];    // i
    kernel->dim[1]  = buffer[14];   // o
    kernel->dim[2]  = buffer[6];    // h
    kernel->dim[3]  = buffer[7];    // w
    kernel->dim_count = 4;
    weight_size = kernel->dim[0] * kernel->dim[1] *  kernel->dim[2] *  kernel->dim[3];
    /* get kernel min max */
    find_min_max((float *)kernel->data, &max_value, &min_value, weight_size);
    kernel->qinfo->min = min_value;
    kernel->qinfo->max = max_value;
    kernel->name = "kernel";


    struct csi_tensor *bias  = csi_alloc_tensor(sess);
    bias->data = (float *)(buffer + 17 + in_size + weight_size);
    bias->dim[0] = buffer[14];
    bias->dim_count = 1;
    /* get bias min max */
    find_min_max((float *)bias->data, &max_value, &min_value, kernel->dim[0]);
    bias->qinfo->min = min_value;
    bias->qinfo->max = max_value;
    bias->name = "bias";


    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim[0]  = buffer[0];         // batch
    output->dim[1]  = buffer[14];        // out_channel
    output->dim[2]  = buffer[16];        // height
    output->dim[3]  = buffer[15];        // width
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];

    reference->data = (float *)(buffer + 17 + in_size + weight_size + bias->dim[0]);
    /* get output min max */
    find_min_max((float *)reference->data, &max_value, &min_value, out_size);
    output->qinfo->min = min_value;
    output->qinfo->max = max_value;
    output->name = "output";


    struct conv2d_params params;
    params.stride_height = buffer[4];
    params.stride_width  = buffer[5];
    params.pad_left   = buffer[8];
    params.pad_right  = buffer[9];
    params.pad_top    = buffer[10];
    params.pad_down   = buffer[11];
    params.dilation_width  = buffer[12];
    params.dilation_height = buffer[13];
    params.group      = 1;
    params.base.api = CSINN_API;
    params.base.layout = CSINN_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.base.name = "params";

    /*
        when input_data layout = NCHW, kernel_layout = OIHW, actually, kernel_data need IOHW layout  ✔
                                       kernel_layout = IOHW, kernel_data need OIHW layout
    */
    if (csi_deconv2d_init(input, output, kernel, bias, &params) != CSINN_TRUE) {
        printf("deconv2d init fail.\n\t");
        return -1;
    }

    csi_pnna_input_setup(input, sess);
    csi_set_input(0, input, sess);

    csi_deconv2d(input, output, kernel, bias, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    struct csi_tensor *input_tensor = csi_alloc_tensor(NULL);
    input_tensor->data = input_data;
    csi_update_input(0, input_tensor, sess);
    csi_session_run(sess);

    struct csi_tensor *output_tensor = csi_alloc_tensor(NULL);
    output_tensor->is_const = 0;
    int output_num = csi_get_output_number(sess);
    printf("output_num = %d\n", output_num);
    csi_get_output(0, output_tensor, sess);    // output_num = 1

    /* FIX ME */
    float difference = argc > 2 ? atof(argv[2]) : 1e-5;
    result_verify_f32(reference->data, output_tensor->data, input->data, difference, out_size, false);

    /* evaluate error by kl and cosine similarity */
    float *output_tensor_data = (float *)output_tensor->data;
    float kl = compute_kl(output_tensor_data, reference->data, out_size);
    printf("The kl diver is %f.\n", kl);
    float cs = compute_cs(output_tensor_data, reference->data, out_size);
    printf("The cos sim is %f.\n", cs);

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
