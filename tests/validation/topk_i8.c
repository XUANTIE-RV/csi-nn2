/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "csi_nn.h"
#include "test_utils.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of topk i8.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output1 = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output2 = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference1 = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference2 = csinn_alloc_tensor(NULL);
    struct csinn_topk_params *params = csinn_alloc_params(sizeof(struct csinn_topk_params), NULL);
    int in_size = 1, out_size = 1;
    float error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    params->k = buffer[0];
    input->dim_count = buffer[1];
    output1->dim_count = input->dim_count;
    output2->dim_count = input->dim_count;
    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 2];
        output1->dim[i] = input->dim[i];
        output2->dim[i] = input->dim[i];
        in_size *= input->dim[i];
    }

    out_size = in_size / input->dim[input->dim_count - 1] * params->k;
    input->dtype = CSINN_DTYPE_INT8;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    output1->dtype = CSINN_DTYPE_INT8;
    output1->layout = CSINN_LAYOUT_NCHW;
    output1->is_const = 0;
    output1->quant_channel = 1;

    output2->dtype = CSINN_DTYPE_INT32;
    output2->layout = CSINN_LAYOUT_NCHW;
    output2->is_const = 0;
    output2->quant_channel = 1;

    params->base.api = CSINN_API;

    float *src_in_data = (float *)(buffer + 2 + input->dim_count);
    float *ref_data1 = (float *)(buffer + 2 + input->dim_count + in_size);
    int *ref_data2 = (int *)(buffer + 2 + input->dim_count + in_size + out_size);

    int8_t *input_data = (int8_t *)malloc(in_size * sizeof(int8_t));

    input->data = src_in_data;
    get_quant_info(input);

    for (int i = 0; i < in_size; i++) {
        input_data[i] = shl_ref_quantize_f32_to_i8(src_in_data[i], input->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_i8_to_f32(input_data[i], input->qinfo);
        if (isinf(src_in_data[i]) && isinf(output_tmp) ||
            isnan(src_in_data[i]) && isnan(output_tmp)) {
            continue;
        } else {
            error1 = fabs(src_in_data[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(src_in_data[i] - output_tmp) / fabs(src_in_data[i] + 1e-9);
            }
        }
        if (error1 > error) {
            error = error1;
        }
    }
    // if (input->dim_count == 1 && params->k == 1) Follow the input scale and zero_point
    if (input->dim_count != 1 || params->k != 1) {
        output1->data = ref_data1;
        get_quant_info(output1);
    } else {
        output1->qinfo = input->qinfo;
    }

    input->data = input_data;
    reference1->data = ref_data1;
    reference2->data = ref_data2;
    output1->data = (int8_t *)malloc(out_size * sizeof(int8_t));
    output2->data = (int *)malloc(out_size * sizeof(int));

    float difference1 = argc > 2 ? atof(argv[2]) : 2 * error;
    float difference2 = argc > 3 ? atof(argv[3]) : 0;
    printf("The max error is %.6lf.\n", error);

    if (csinn_topk_init(input, output1, output2, params) == CSINN_TRUE) {
        csinn_topk(input, output1, output2, params);
    }

    result_verify_8(reference1->data, output1, input->data, difference1, out_size, false);
    /*
    when inputs: such as [5.0001, 5.0000]
    they all quantized by [200, 200]
    so their output_indices are reversed
    */
    // result_verify_int32(reference2->data, output2->data, input->data, difference2, out_size,
    // false);

    free(buffer);
    free(output1->data);
    free(output2->data);
    free(input_data);
    return done_testing();
}
