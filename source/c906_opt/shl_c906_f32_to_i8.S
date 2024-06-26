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

/**************************************************************************************************

    void shl_c906_f32_to_i8(const float *input,
                            int8_t *output,
                            int32_t offset,
                            float *scale,
                            uint32_t length)

    Algorithm works as follows:
        (1)

    register definition:
        a0: input addr
        a1: output addr
        a2: offset
        a3: scale point
        a4: element length

 *************************************************************************************************/
#define SHL_C906_F32_TO_8 shl_c906_f32_to_i8
#define SHL_C906_F32_TO_8_SIGNED

    .file           "shl_c906_f32_to_i8.S"
    .section        .text.SHL_C906_F32_TO_8, "ax", @progbits
    .align          5
    .global         SHL_C906_F32_TO_8
    .type           SHL_C906_F32_TO_8, @function

#include "./shl_c906_f32_to_u8.S"

#undef SHL_C906_F32_TO_8_SIGNED
