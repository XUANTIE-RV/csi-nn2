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
    init_testsuite("Testing function of batch normalization(graph).\n");

    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_LIGHT;
    sess->base_dtype = CSINN_DTYPE_INT8;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    float min_value, max_value;
    int in_size = 0, out_size = 0;

    int *buffer = read_input_data_f32(argv[1]);
    int channel_size = buffer[4];

    // ------------------------ input ----------------------
    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim[0] = buffer[1];          // batch
    input->dim[1] = buffer[4];          // channel
    input->dim[2] = buffer[2];          // height
    input->dim[3] = buffer[3];          // width
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];

    float *input_data = (float *)(buffer + 6);
    /* get input min max */
    find_min_max((float *)input_data, &max_value, &min_value, in_size);
    input->qinfo->min = min_value;
    input->qinfo->max = max_value;
    input->name = "input";

    // ------------------------ mean ----------------------
    struct csi_tensor *mean  = csi_alloc_tensor(sess);
    mean->data = (float *)(buffer + 6 + in_size);
    mean->dim[0] = channel_size;
    mean->dim_count = 1;
    /* get mean min max */
    find_min_max((float *)mean->data, &max_value, &min_value, channel_size);
    mean->qinfo->min = min_value;
    mean->qinfo->max = max_value;
    mean->name = "mean";

    // ------------------------ variance ----------------------
    struct csi_tensor *variance  = csi_alloc_tensor(sess);
    variance->data = (float *)(buffer + 6 + in_size + channel_size);
    variance->dim[0] = channel_size;
    variance->dim_count = 1;
    /* get variance min max */
    find_min_max((float *)variance->data, &max_value, &min_value, channel_size);
    variance->qinfo->min = min_value;
    variance->qinfo->max = max_value;
    variance->name = "variance";

    // ------------------------ gamma ----------------------
    struct csi_tensor *gamma  = csi_alloc_tensor(sess);
    gamma->data = (float *)(buffer + 6 + in_size + channel_size * 2);
    gamma->dim[0] = channel_size;
    gamma->dim_count = 1;
    /* get gamma min max */
    find_min_max((float *)gamma->data, &max_value, &min_value, channel_size);
    gamma->qinfo->min = min_value;
    gamma->qinfo->max = max_value;
    gamma->name = "gamma";


    // ------------------------ beta ----------------------
    struct csi_tensor *beta  = csi_alloc_tensor(sess);
    beta->data = (float *)(buffer + 6 + in_size + channel_size * 3);
    beta->dim[0] = channel_size;
    beta->dim_count = 1;
    /* get gamma min max */
    find_min_max((float *)beta->data, &max_value, &min_value, channel_size);
    beta->qinfo->min = min_value;
    beta->qinfo->max = max_value;
    beta->name = "beta";

    // ------------------------ output ----------------------
    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim[0] = input->dim[0];
    output->dim[1] = input->dim[1];
    output->dim[2] = input->dim[2];
    output->dim[3] = input->dim[3];
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];

    reference->data = (float *)(buffer + 6 + in_size + channel_size * 4);
    /* get output min max */
    find_min_max((float *)reference->data, &max_value, &min_value, out_size);
    output->qinfo->min = min_value;
    output->qinfo->max = max_value;
    output->name = "output";


    struct bn_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.epsilon = *((float *)buffer + 5);


    if (csi_batch_normalization_init(input, mean, variance, gamma, beta, output, &params) != CSINN_TRUE) {
        printf("batch normalization init fail.\n\t");
        return -1;
    }

    csi_pnna_input_setup(input, sess);
    csi_set_input(0, input, sess);

    csi_batch_normalization(input, mean, variance, gamma, beta, output, &params);

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
    csi_get_output(0, output_tensor, sess);

    /* FIX ME */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
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
