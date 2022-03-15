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
#include "./valid_data/fully_data_q7.dat"


static void verify_fullyconnected_q7(void *input_data,
                                     void *weight_data,
                                     void *bias_data,
                                     void *ref_data,
                                     uint16_t in_nodes,
                                     uint16_t out_nodes,
                                     uint16_t bias_shift,
                                     uint16_t out_shift,
                                     float difference)
{
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size, out_size, weight_size = 0, bias_size = 0;

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = in_nodes;
    input->dim_count = 2;
    input->dtype = CSINN_DTYPE_INT8;
    input->name = "input";
    in_size = input->dim[0] * input->dim[1];


    struct csi_tensor *weight = csi_alloc_tensor(NULL);
    weight->dim[0] = out_nodes;
    weight->dim[1] = in_nodes;
    weight->dim_count = 2;
    weight->dtype = CSINN_DTYPE_INT8;
    weight->name = "weight";
    weight_size = weight->dim[0] * weight->dim[1];


    struct csi_tensor *bias = csi_alloc_tensor(NULL);
    bias->dim[0] = out_nodes;
    bias->dim_count = 1;
    bias->dtype = CSINN_DTYPE_INT8;
    bias->name = "bias";
    bias_size = bias->dim[0];
    bias->qinfo->shift = bias_shift;

    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = out_nodes;
    output->dim_count = 2;
    output->dtype = CSINN_DTYPE_INT8;
    output->name = "output";
    out_size = output->dim[0] * output->dim[1];
    output->qinfo->shift = out_shift;

    struct fc_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_LAYER;
    params.units = out_nodes;

    input->data      = (uint8_t *)input_data;
    weight->data     = (uint8_t *)weight_data;
    bias->data       = (uint8_t *)bias_data;
    reference->data  = (uint8_t *)ref_data;
    uint8_t *output_tmp = (uint8_t *)malloc(out_size);
    output->data     = output_tmp;

    if (csi_fullyconnected_init(input, output, weight, bias, &params) == CSINN_TRUE) {
        csi_fullyconnected(input, output, weight, bias, &params);
    }

    result_verify_q7(reference->data, output->data, input->data, difference, out_size, false);
    free(output_tmp);
    free(input);
    free(weight);
    free(bias);
    free(output);
    free(reference);
}


int main(int argc, char** argv)
{
    init_testsuite("Testing function of fullyconnected q7 for xt800.\n");

    verify_fullyconnected_q7(fully_connect_input_3, fully_connect_weight_3, fully_connect_bias_3, fully_connect_result_6,
                             256, 128, 0, 8, 0.0f);

    verify_fullyconnected_q7(fully_connect_input_4, fully_connect_weight_4, fully_connect_bias_4, fully_connect_result_7,
                             256, 64, 0, 10, 0.0f);

    verify_fullyconnected_q7(fully_connect_input_5, fully_connect_weight_5, fully_connect_bias_5, fully_connect_result_8,
                             128, 128, 0, 12, 0.0f);

    /* leftover test */
    verify_fullyconnected_q7(fully_connect_input_3, fully_connect_weight_3, fully_connect_bias_3, fully_connect_result_9,
                             255, 127, 0, 8, 0.0f);

    verify_fullyconnected_q7(fully_connect_input_4, fully_connect_weight_4, fully_connect_bias_4, fully_connect_result_10,
                             255, 63, 0, 10, 0.0f);

    verify_fullyconnected_q7(fully_connect_input_5, fully_connect_weight_5, fully_connect_bias_5, fully_connect_result_11,
                             127, 127, 0, 12, 0.0f);
}
