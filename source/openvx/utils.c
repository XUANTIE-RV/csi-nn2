/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#include "csi_utils.h"


int32_t csi_get_ceil_mode_fix(int32_t input, int32_t kernel, int32_t stride, int32_t pad)
{
    int output0 = (input + pad * 2 - kernel + stride - 1) / stride + 1;
    int output1 = (input + pad * 2 - kernel) / stride + 1;

    if (output0 == output1) {
        return 0;
    } else {
        return 1;
    }
}
