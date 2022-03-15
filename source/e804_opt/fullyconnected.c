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

#include "csi_e804.h"


int csi_e804_fullyconnected_q7(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct csi_tensor *weights,
                               struct csi_tensor *bias,
                               struct fc_params *params)
{
    q7_t *input_data = (q7_t *)input->data;
    q7_t *weight_data = (q7_t *)weights->data;
    q7_t *bias_data = (q7_t *)bias->data;
    q7_t *output_data = (q7_t *)output->data;

    csky_dsp2_fully_connected_q7(input_data, weight_data, input->dim[1], weights->dim[0],
                                 bias->qinfo->shift, output->qinfo->shift, bias_data, output_data);
    return CSINN_TRUE;
}

int csi_e804_fullyconnected_q15(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *weights,
                                struct csi_tensor *bias,
                                struct fc_params *params)
{
    q15_t *input_data = (q15_t *)input->data;
    q15_t *weight_data = (q15_t *)weights->data;
    q15_t *bias_data = (q15_t *)bias->data;
    q15_t *output_data = (q15_t *)output->data;

    csky_dsp2_fully_connected_q15(input_data, weight_data, input->dim[1], weights->dim[0],
                                  bias->qinfo->shift, output->qinfo->shift, bias_data, output_data);
    return CSINN_TRUE;
}
              