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

void op_test_run(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                 struct diso_params *params, struct csi_session *sess,
                 struct csi_tensor *real_input0, struct csi_tensor *real_input1, float *output_data,
                 float diff);

void test_u8_asym(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                  struct diso_params *params, float difference)
{
    printf("test add u8 asym\n");
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    sess->base_quant_type = CSINN_QUANT_UINT8_ASYM;
    // sess->debug_level = CSI_DEBUG_LEVEL_INFO;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_UINT8;

    struct csi_tensor *qinput0 = convert_f32_input(input0, test_dtype, sess);
    struct csi_tensor *qinput1 = convert_f32_input(input1, test_dtype, sess);
    struct csi_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csi_tensor *real_input0 = convert_f32_input(input0, test_dtype, sess);
    struct csi_tensor *real_input1 = convert_f32_input(input1, test_dtype, sess);
    op_test_run(qinput0, qinput1, qoutput, params, sess, real_input0, real_input1, output->data,
                difference);
}

void test_add(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
              struct diso_params *params, float difference)
{
    params->base.api = CSINN_ANOLE;

    test_u8_asym(input0, input1, output, params, difference);
}
