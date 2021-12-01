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

/* CSI-NN2 version 1.8.x */

#include "csi_c906.h"

int csi_c906_global_avgpool_f32(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct pool_params *params)
{
    float *input_data  = (float *)input->data;
    float *output_data = (float *)output->data;

    int batch = input->dim[0];
    int in_c  = input->dim[1];
    int in_h  = input->dim[2];
    int in_w  = input->dim[3];
    int in_hw = in_h * in_w;

    int in_hw8 = in_hw >> 3;
    int in_hw_tail = in_hw & 7;

    for(int b = 0; b < batch; b++) {

        for(int c = 0; c < in_c; c++) {

            float sum = 0.0f;
#if __riscv_vector == 128
            if(in_hw8 > 0) {
                asm volatile(
                    "vsetvli        zero, zero, e32, m1\n\t"
                    "mv             t0, %1\n\t"
                    "vfmv.v.f       v2, %2\n\t"     // init res v2[0..3] = 0.0f
                    "vfmv.s.f       v3, %2\n\t"     // init tmp v3[0] = 0.0f
                "2:\n\t"
                    // "vlseg2e.v      v0, (%0)\n\t"   // v0[0..3] = line0[0,2.4.6]   v1[0..3] = line0[1,3,5,7]
                    // "vfadd.vv       v2, v2, v0\n\t"
                    // "vfadd.vv       v2, v2, v1\n\t"
                    // "addi           %0, %0, 32\n\t"

                    "vlw.v          v0, (%0)\n\t"
                    "addi           %0, %0, 16\n\t"
                    "vfadd.vv       v2, v0, v2\n\t"

                    "vlw.v          v1, (%0)\n\t"
                    "addi           %0, %0, 16\n\t"
                    "vfadd.vv       v2, v1, v2\n\t"

                    "addi           t0, t0, -1\n\t"
                    "bnez           t0, 2b\n\t"

                    "vfredsum.vs    v3, v2, v3\n\t"     // v3[0] = unorder_sum(v2[0..3]) + v3[0]
                    "vfmv.f.s       %2, v3\n\t"         // sum = v3[0]
                    :"=r"(input_data),  // %0
                    "=r"(in_hw8),       // %1
                    "=f"(sum)           // %2
                    :"0"(input_data),
                    "1"(in_hw8),
                    "2"(sum)
                    :"cc", "memory", "v0", "v1", "v2", "v3", "t0"
                );
            }
#else
            float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for(int i = 0; i < in_hw8; i++) {
                tmp[0] += input_data[0] + input_data[1];
                tmp[1] += input_data[2] + input_data[3];
                tmp[2] += input_data[4] + input_data[5];
                tmp[3] += input_data[6] + input_data[7];
                input_data += 8;
            }
            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif  //__riscv_vector

            for(int i = 0; i < in_hw_tail; i++) {
                sum += input_data[i];
            }
            input_data += in_hw_tail;
            output_data[0] = (float)sum / in_hw;
            output_data++;
        }
    }
    return CSINN_TRUE;
}
