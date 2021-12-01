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

#include "csi_nn.h"
#include "math_snr.h"
#include "test_utils.h"

void op_test_run(struct csi_tensor *input, struct csi_tensor *kernel, struct csi_tensor *bias,
                 struct csi_tensor *output, struct conv2d_params *params, struct csi_session *sess,
                 struct csi_tensor *real_input, float *output_data, float diff);

void test_f16(struct csi_tensor *input, struct csi_tensor *kernel, struct csi_tensor *bias,
              struct csi_tensor *output, struct conv2d_params *params, float difference)
{
    printf("test conv2d f16\n");
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_C906;
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->base_dtype = CSINN_DTYPE_FLOAT16;
    sess->base_quant_type = CSINN_QUANT_FLOAT16;
    // sess->debug_level = CSI_DEBUG_LEVEL_INFO;
    params->base.sess = sess;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_FLOAT16;

    struct csi_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csi_tensor *qkernel = convert_f32_input(kernel, test_dtype, sess);
    struct csi_tensor *qbias = convert_f32_input(bias, test_dtype, sess);
    struct csi_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csi_tensor *real_input = convert_f32_input(input, test_dtype, sess);

    op_test_run(qinput, qkernel, qbias, qoutput, params, sess, real_input, output->data,
                difference);
}

void test_f32(struct csi_tensor *input, struct csi_tensor *kernel, struct csi_tensor *bias,
              struct csi_tensor *output, struct conv2d_params *params, float difference)
{
    printf("test conv2d f32\n");
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_C906;
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->base_quant_type = CSINN_QUANT_FLOAT32;
    sess->base_dtype = CSINN_DTYPE_FLOAT32;
    // sess->debug_level = CSI_DEBUG_LEVEL_INFO;
    params->base.sess = sess;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_FLOAT32;

    struct csi_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csi_tensor *qkernel = convert_f32_input(kernel, test_dtype, sess);
    struct csi_tensor *qbias = convert_f32_input(bias, test_dtype, sess);
    struct csi_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csi_tensor *real_input = convert_f32_input(input, test_dtype, sess);

    op_test_run(qinput, qkernel, qbias, qoutput, params, sess, real_input, output->data,
                difference);
}

void test_conv2d(struct csi_tensor *input, struct csi_tensor *kernel, struct csi_tensor *bias,
                  struct csi_tensor *output, struct conv2d_params *params, float difference)
{
    params->base.api = CSINN_C906;
    params->base.run_mode = CSINN_RM_CPU_GRAPH;
    test_f16(input, kernel, bias, output, params, difference);
    test_f32(input, kernel, bias, output, params, difference);
}

