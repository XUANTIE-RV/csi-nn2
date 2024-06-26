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

#include "shl_gref.h"

int shl_gref_rms_norm(struct csinn_tensor *input, struct csinn_tensor *weights,
                      struct csinn_tensor *output, struct csinn_rms_norm_params *params)
{
    shl_gref_diso_op(input, weights, output, CSINN_OP_RMS_NORM, params);
    return CSINN_TRUE;
}

int shl_gref_rms_norm_infer_shape(struct csinn_tensor *input, struct csinn_tensor *weights,
                                  struct csinn_tensor *output, struct csinn_rms_norm_params *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    output->dim_count = input->dim_count;
    for (int i = 0; i < input->dim_count; i++) {
        output->dim[i] = input->dim[i];
    }

    SHL_DEBUG_CALL(shl_rms_norm_debug_info(input, weights, output, params, __func__));
    return CSINN_TRUE;
}
