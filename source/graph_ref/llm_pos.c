/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
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

int shl_gref_llm_pos(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_llm_pos_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_LLM_POS, params);
    return CSINN_TRUE;
}

int shl_gref_llm_pos_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_llm_pos_params *params)
{
    if (params->mode == CSINN_LLM_POS_CACHE_COPY_IN) {
        // do nothing
    } else if (params->mode == CSINN_LLM_POS_CACHE_COPY_OUT) {
        output->dim_count = 4;
        output->dim[0] = 1;
        output->dim[1] = params->pos[0] + params->seqlen;
        output->dim[2] = 32;
        output->dim[3] = 128;
    } else if (params->mode == CSINN_LLM_POS_MASK) {
        output->dim_count = input->dim_count;
        for (int i = 0; i < input->dim_count; i++) {
            output->dim[i] = input->dim[i];
        }
    }
    SHL_DEBUG_CALL(shl_llm_pos_debug_info(input, output, params, __func__));
    return CSINN_TRUE;
}
