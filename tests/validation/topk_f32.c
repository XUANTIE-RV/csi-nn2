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
    init_testsuite("Testing function of topk f32.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output1 = csi_alloc_tensor(NULL);
    struct csi_tensor *output2 = csi_alloc_tensor(NULL);
    struct csi_tensor *reference1 = csi_alloc_tensor(NULL);
    struct csi_tensor *reference2 = csi_alloc_tensor(NULL);
    struct topk_params params;
    int in_size = 1, out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    params.k = buffer[0];
    input->dim_count = buffer[1];
    output1->dim_count = input->dim_count;
    output2->dim_count = input->dim_count;
    for(int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 2];
        output1->dim[i] = input->dim[i];
        output2->dim[i] = input->dim[i];
        in_size *= input->dim[i];
    }
    output1->dim[output1->dim_count - 1] = params.k;    // values last dim = k
    output2->dim[output2->dim_count - 1] = params.k;    // indices last dim = k

    out_size = in_size / input->dim[input->dim_count - 1] * params.k;
    input->dtype = CSINN_DTYPE_FLOAT32;
    output1->dtype = CSINN_DTYPE_FLOAT32;
    output2->dtype = CSINN_DTYPE_INT32;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    input->data     = (float *)(buffer + 2 + input->dim_count);
    reference1->data = (float *)(buffer + 2 + input->dim_count + in_size);
    reference2->data = (int *)(buffer + 2 + input->dim_count + in_size + out_size);

    output1->data   = (float *)malloc(out_size * sizeof(float));
    output2->data   = (int *)malloc(out_size * sizeof(int));
    float difference1 = argc > 2 ? atof(argv[2]) : 1e-6;
    float difference2 = argc > 3 ? atof(argv[3]) : 0;

    if (csi_topk_init(input, output1, output2, &params) == CSINN_TRUE) {
        csi_topk(input, output1, output2, &params);
    }

    result_verify_f32((float *)reference1->data, output1->data, input->data, difference1, out_size, false);
    result_verify_int32(reference2->data, output2->data, input->data, difference2, out_size, false);

    free(buffer);
    free(output1->data);
    free(output2->data);
    return done_testing();
}
