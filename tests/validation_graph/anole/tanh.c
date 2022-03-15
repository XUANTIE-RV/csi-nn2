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

void op_test_run(struct csi_tensor *input, struct csi_tensor *output, struct siso_params *params,
                 struct csi_session *sess, struct csi_tensor *real_input, float *output_data,
                 float diff);

void test_u8_asym(struct csi_tensor *input, struct csi_tensor *output, struct siso_params *params,
                 float difference)
{
    printf("test tanh u8 asym\n");
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    sess->base_quant_type = CSINN_QUANT_UINT8_ASYM;
    // sess->debug_level = CSI_DEBUG_LEVEL_INFO;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_UINT8;

    struct csi_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    /* output scale zp equal to input */
    struct csi_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csi_tensor *real_input = convert_f32_input(input, test_dtype, sess);

    op_test_run(qinput, qoutput, params, sess, real_input, output->data, difference);
}

void test_tanh(struct csi_tensor *input, struct csi_tensor *output, struct siso_params *params,
               float difference)
{
    params->base.api = CSINN_ANOLE;

    test_u8_asym(input, output, params, difference);

}
