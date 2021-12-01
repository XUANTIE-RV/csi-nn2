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
    init_testsuite("Testing function of fullyconnected u8.\n");

    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    sess->base_dtype = CSINN_DTYPE_UINT8;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct csi_tensor *weight = csi_alloc_tensor(NULL);
    struct csi_tensor *bias = csi_alloc_tensor(NULL);
    struct fc_params params;
    int in_size0, in_size1, out_size;
    int zp, quantized_multiplier, shift;
    float max_value, min_value, scale, scale1, scale2, scale3;


    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0]  = buffer[0];          // batch
    input->dim[1]  = buffer[1];          // in_size
    weight->dim[0] = buffer[2];          // out_size
    weight->dim[1] = buffer[1];          // in_size
    bias->dim[0]   = buffer[2];
    output->dim[0] = buffer[0];
    output->dim[1] = buffer[2];
    input->dim_count  = 2;
    weight->dim_count = 2;
    bias->dim_count   = 1;
    output->dim_count = 2;
    in_size0 = input->dim[0] * input->dim[1];
    in_size1 = weight->dim[0] * weight->dim[1];
    out_size = output->dim[0] * output->dim[1];
    input->dtype = CSINN_DTYPE_UINT8;
    weight->dtype = CSINN_DTYPE_UINT8;
    bias->dtype = CSINN_DTYPE_UINT8;    // FIX ME
    output->dtype = CSINN_DTYPE_UINT8;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.base.name = "params";
    params.units = buffer[2];   // out_nodes


    float *src_in   = (float *)(buffer + 3);
    float *weight_in   = (float *)(buffer + 3 + in_size0);
    float *bias_in   = (float *)(buffer + 3 + in_size0 + in_size1);
    float *ref   = (float *)(buffer + 3 + in_size0 + in_size1 + buffer[2]);

    uint8_t *input_tmp = malloc(in_size0 * sizeof(char));
    uint8_t *weight_tmp = malloc(in_size1 * sizeof(char));
    int32_t *bias_tmp = (int32_t *)malloc(buffer[2] * sizeof(int32_t));

    input->qinfo = get_quant_info(src_in, in_size0);
    scale1 = input->qinfo->scale;

    for(int i = 0; i < in_size0; i++) {
        input_tmp[i] = csi_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }

    input->name = "input";

    weight->qinfo = get_quant_info(weight_in, in_size1);
    scale2 = weight->qinfo->scale;

    for(int i = 0; i < in_size1; i++) {
        weight_tmp[i] =  csi_ref_quantize_f32_to_u8(weight_in[i], weight->qinfo);
    }
    weight->name = "weight";
    weight->is_const = 1;



    scale=scale1*scale2;
    for(int i = 0; i < buffer[2]; i++) {
        bias_tmp[i] = (int32_t)(bias_in[i]/scale);
    }
    bias->is_const = 1;

    output->qinfo = get_quant_info(ref, out_size);
    scale3=output->qinfo->scale; 
    scale=(scale1*scale2)/scale3;
    quantize_multiplier(scale, &quantized_multiplier, &shift);
    output->qinfo->multiplier = quantized_multiplier;
    output->qinfo->shift      = shift;

    weight->data    = weight_tmp;
    bias->data  = bias_tmp;

    reference->data = ref;


    if (csi_fullyconnected_init(input, output, weight, bias, &params) != CSINN_TRUE) {
        printf("fullyconnected init fail.\n\t");
        return -1;
    }
 
    csi_ovx_set_tensor(input, sess);
    csi_set_input(0, input, sess);

    csi_fullyconnected(input, output, weight, bias, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    struct csi_tensor *input_tensor = csi_alloc_tensor(NULL);
    input_tensor->data = input_tmp;
    csi_update_input(0, input_tensor, sess);
    csi_session_run(sess);

    struct csi_tensor *output_tensor = csi_alloc_tensor(NULL);
    output_tensor->is_const = 0;
    int output_num = csi_get_output_number(sess);
    printf("output_num = %d\n", output_num);
    csi_get_output(0, output_tensor, sess);    // output_num = 1


    quantize_multiplier(scale3, &quantized_multiplier, &shift);
    output_tensor->qinfo->multiplier = quantized_multiplier;
    output_tensor->qinfo->shift      = shift; 
    output_tensor->qinfo->zero_point = output->qinfo->zero_point;
    output_tensor->dtype == CSINN_DTYPE_UINT8;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    result_verify_8(reference->data, output_tensor, input->data, difference, out_size, false);

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
