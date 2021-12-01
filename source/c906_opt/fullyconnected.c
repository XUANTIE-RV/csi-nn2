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

#include "csi_c906.h"

/*
    change memory layout for weight matrix [out_nodes * in_nodes] by N shape
*/
void csi_c906_reorder_weight_fp16(__fp16 *src, __fp16 *dst, int m, int k, int ldx)
{
    int i = 0;
    for (; i + 7 < m; i += 8) {
        for (int j = 0; j < k; j++) {
            dst[i * k + 8 * j + 0] = src[(i + 0) * k + j];
            dst[i * k + 8 * j + 1] = src[(i + 1) * k + j];
            dst[i * k + 8 * j + 2] = src[(i + 2) * k + j];
            dst[i * k + 8 * j + 3] = src[(i + 3) * k + j];
            dst[i * k + 8 * j + 4] = src[(i + 4) * k + j];
            dst[i * k + 8 * j + 5] = src[(i + 5) * k + j];
            dst[i * k + 8 * j + 6] = src[(i + 6) * k + j];
            dst[i * k + 8 * j + 7] = src[(i + 7) * k + j];
        }
    }
    dst += i * k;
    src += i * k;
    for (; i < m; i++) {
        csi_c906_memcpy(dst, src, sizeof(__fp16) * ldx);
        dst += k;
        src += k;
    }
}


void csi_c906_fc_gemv_transform_weight_fp16(struct csi_tensor *weights)
{
    __fp16 *weight_data = (__fp16 *)weights->data;

    int n = weights->dim[0];        // out_nodes
    int k = weights->dim[1];        // in_nodes

    __fp16* pa_reorder = (__fp16 *)csi_mem_alloc(n * k * sizeof(__fp16));
    csi_c906_reorder_weight_fp16(weight_data, pa_reorder, n, k, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(__fp16));
    csi_mem_free(pa_reorder);
}


int csi_c906_fullyconnected_f32(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *weights,
                                struct csi_tensor *bias,
                                struct fc_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *weights_data = weights->data;
    float *bias_data = bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    const int batches = output->dim[0];
    const int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    const int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    float zero = 0.0f;
    asm volatile(
                "mv             a0, %5\n\t"
            "1:\n\t"
                "mv             a1, %6\n\t"
                "2:\n\t"
                    "mv             a2, %7\n\t"
                    "vxor.vv        v8, v8, v8\n\t"     // clear
                    "3:\n\t"
                        "vsetvli        t0, a2, e32, m1\n\t"
                        "vlw.v          v2, (%2)\n\t"       // load input_data
                        "sub            a2, a2, t0\n\t"
                        "slli           t0, t0, 2\n\t"
                        "add            %2, %2, t0\n\t"     // bump input_data pointer
                        "vlw.v          v4, (%3)\n\t"       // load weight_data
                        "add            %3, %3, t0\n\t"     // bump weight_data pointer
                        "vfsub.vv       v6, v6, v6\n\t"     // clear v6
                        "vfmacc.vv      v6, v2, v4\n\t"
                        "vfredsum.vs    v8, v6, v8\n\t"     // v8[0] = v8[0] + sum(v6[0..i])
                        "bnez           a2, 3b\n\t"

                    "vfmv.f.s       ft1, v8\n\t"
                    "beqz           %8, 4f\n\t"         // bias_dims_count = 0 (bias = NULL) ?

                    "flw            ft0, 0(%4)\n\t"     // load bias_data
                    "addi           %4, %4, 4\n\t"      // bump bias_data pointer
                    "fadd.s         ft1, ft1, ft0\n\t"

                "4:\n\t"
                    "fsw            ft1, 0(%0)\n\t"     // store output_data
                    "addi           %0, %0, 4\n\t"      // bump output_data pointer

                    "slli           a3, %7, 2\n\t"
                    "sub            %2, %2, a3\n\t"
                    "addi           a1, a1, -1\n\t"
                    "bnez           a1, 2b\n\t"

                "add            %2, %2, a3\n\t"
                "mul            t1, %6, %7\n\t"
                "slli           t1, t1, 2\n\t"
                "sub            %3, %3, t1\n\t"     // finish all output_nodes, jump weights_data pointer
                "slli           t2, %6, 2\n\t"
                "sub            %4, %4, t2\n\t"     // finish all output_nodes, jump bias_data pointer

                "addi           a0, a0, -1\n\t"
                "bnez           a0, 1b\n\t"

                :"=r"(output_data)  // %0
                :"0"(output_data),  // %1
                "r"(input_data),    // %2
                "r"(weights_data),  // %3
                "r"(bias_data),     // %4
                "r"(batches),       // %5
                "r"(output_depth),  // %6
                "r"(accum_depth),   // %7
                "r"(bias_dims_count)// %8
                : "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "ft0", "ft1"
    );

    return CSINN_TRUE;
}

int csi_c906_fullyconnected_fp16(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct csi_tensor *weights,
                                 struct csi_tensor *bias,
                                 struct fc_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = (__fp16 *)weights->data;
    __fp16 *bias_data = (__fp16 *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = output->dim[0];
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    for (int b = 0; b < batches; b++) {

        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;

        asm volatile(
            "vsetvli        zero, zero, e16, m1\n\t"    // set vl = 8

            "srai           t1, %5, 3\n\t"  // t1 = in_node >> 3 (k8)
            "andi           t2, %5, 7\n\t"  // t2 = in_node & 7
            "srai           t2, t2, 2\n\t"  // t2 = (in_node & 7) >> 2 (k4)
            "andi           t3, %5, 3\n\t"  // t3 = in_node & 3
            "srai           t3, t3, 1\n\t"  // t3 = (in_node & 3) >> 1 (k2)
            "andi           t4, %5, 1\n\t"  // t4 = in_node & 1 (k1)

            "slli           s2, %5, 1\n\t"  // load stride = in_node * 2
            "slli           s3, %5, 4\n\t"  // 8 lines weight_data (in_node * 8 * 2)

            "srai           t0, %4, 3\n\t"  // out_node >> 3 (n8)
            "beqz           t0, 7f\n\t"     // jump to m1n4

        "0:\n\t"    // m1n8

            "vmv.v.x        v24, zero\n\t"  // clear
            "beqz           %3, 1f\n\t"     // if bias_data = NULL (bias->dim_count = 0)
            "vle.v          v24, (%3)\n\t"  // init out_tmp = bias_data
            "addi           %3, %3, 16\n\t"

            "1:\n\t"
            "mv             a2, %2\n\t"     // a2 = weight_data addr
            "mv             t6, %1\n\t"     // t6 hold input 1 lines start addr

            "vlse.v         v1, (a2), s2\n\t"   // pre-load weight-data
            "addi           a2, a2, 2\n\t"

            "flh            ft0, 0(t6)\n\t"     // pre-load input-data

            "beqz           t1, 3f\n\t"         // if k8 == 0, jump to subkernel_m1n8k4
            "mv             t5, t1\n\t"

            "2:\n\t"
            // m1n8k8
                "vlse.v         v2, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            fa0, 2(t6)\n\t"
                "vfmacc.vf      v24, ft0, v1\n\t"   // 0


                "vlse.v         v3, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            ft0, 4(t6)\n\t"
                "vfmacc.vf      v24, fa0, v2\n\t"   // 1


                "vlse.v         v4, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            fa0, 6(t6)\n\t"
                "vfmacc.vf      v24, ft0, v3\n\t"   // 2


                "vlse.v         v5, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            ft0, 8(t6)\n\t"
                "vfmacc.vf      v24, fa0, v4\n\t"   // 3


                "vlse.v         v6, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            fa0, 10(t6)\n\t"
                "vfmacc.vf      v24, ft0, v5\n\t"   // 4


                "vlse.v         v7, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            ft0, 12(t6)\n\t"
                "vfmacc.vf      v24, fa0, v6\n\t"   // 5


                "vlse.v         v8, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            fa0, 14(t6)\n\t"
                "vfmacc.vf      v24, ft0, v7\n\t"   // 6
                "addi           t6, t6, 16\n\t"     // +8 elements, bump input_data to next k8 addr


                "vlse.v         v1, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            ft0, 0(t6)\n\t"
                "vfmacc.vf      v24, fa0, v8\n\t"   // 7

                "addi           t5, t5, -1\n\t"
                "bnez           t5, 2b\n\t"

            "3:\n\t"
                "beqz           t2, 4f\n\t"     // if k4 == 0, jump to subkernel_m1n8k2
                // m1n8k4
                "vlse.v         v2, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            fa0, 2(t6)\n\t"
                "vfmacc.vf      v24, ft0, v1\n\t"   // 0


                "vlse.v         v3, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            ft0, 4(t6)\n\t"
                "vfmacc.vf      v24, fa0, v2\n\t"   // 1


                "vlse.v         v4, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            fa0, 6(t6)\n\t"
                "vfmacc.vf      v24, ft0, v3\n\t"   // 2
                "addi           t6, t6, 8\n\t"      // +4 elements, bump pa to next k addr


                "vlse.v         v1, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            ft0, 0(t6)\n\t"
                "vfmacc.vf      v24, fa0, v4\n\t"   // 3

            "4:\n\t"
                "beqz           t3, 5f\n\t"     // if k2 == 0, jump to subkernel_m1n8k1
                // m1n8k2
                "vlse.v         v2, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            fa0, 2(t6)\n\t"
                "vfmacc.vf      v24, ft0, v1\n\t"   // 0
                "addi           t6, t6, 4\n\t"      // +2 elements, bump pa to next k addr


                "vlse.v         v1, (a2), s2\n\t"
                "addi           a2, a2, 2\n\t"
                "flh            ft0, 0(t6)\n\t"
                "vfmacc.vf      v24, fa0, v2\n\t"   // 1


            "5:\n\t"
                "beqz           t4, 6f\n\t"     // if k1 == 0, jump to end kernel_m1n8
                // m1n8k1
                "vfmacc.vf      v24, ft0, v1\n\t"

        "6:\n\t"    // end m1n8

            "add            %2, %2, s3\n\t" // weight_data addr + 8 lines

            "vse.v          v24, (%0)\n\t"
            "addi           %0, %0, 16\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 0b\n\t"

        "7:\n\t"    // n_tail
            "andi           t0, %4, 7\n\t"  // n_tail
            "beqz           t0, 13f\n\t"    // if n_tail = 0, jump to ending

            "mv             a2, %2\n\t"     // updata weight_data addr
            "andi           t2, %5, 7\n\t"  // k_tail
            "slli           t3, t2, 1\n\t"  // k_tail * 2

        "8:\n\t"
            "mv             t6, %1\n\t"     // init input_data addr

            "vmv.v.x        v24, zero\n\t"  // clear acc
            "fmv.h.x        ft0, zero\n\t"  // clear
            "beqz           %3, 9f\n\t"     // if bias_data = NULL (bias->dim_count = 0)
            "flh            ft0, 0(%3)\n\t" // else load bias_data
            "addi           %3, %3, 2\n\t"

            "9:\n\t"
            "vfmv.s.f       v25, ft0\n\t"   // v25[0] = bias

            "mv             t5, t1\n\t"     // t5 = k8
            "beqz           t2, 11f\n\t"

            "10:\n\t"
                // m1n1k_tail
                "vsetvli        zero, t2, e16, m1\n\t"
                "vle.v          v1, (t6)\n\t"
                "add            t6, t6, t3\n\t"
                "vle.v          v2, (a2)\n\t"
                "add            a2, a2, t3\n\t"
                "vfmacc.vv      v24, v1, v2\n\t"

                "beqz           t1, 12f\n\t"    // if k8 == 0, jump to end m1n1
                "vsetvli        zero, zero, e16, m1\n\t"

            "11:\n\t"
                // m1n1k8
                "vle.v          v1, (t6)\n\t"
                "addi           t6, t6, 16\n\t"
                "vle.v          v2, (a2)\n\t"
                "addi           a2, a2, 16\n\t"
                "vfmacc.vv      v24, v1, v2\n\t"

                "addi           t5, t5, -1\n\t"
                "bnez           t5, 11b\n\t"

        "12:\n\t"   // end m1n1
            "vfredsum.vs    v25, v24, v25\n\t"  // v25[0] = v25[0](bias) + sum(v24[0..7])
            "vfmv.f.s       fa0, v25\n\t"
            "fsh            fa0, 0(%0)\n\t"
            "addi           %0, %0, 2\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 8b\n\t"

        "13:\n\t"

            :"=r"(init_output), // %0
            "=r"(init_input),   // %1
            "=r"(init_weight),  // %2
            "=r"(init_bias),    // %3
            "=r"(output_depth), // %4
            "=r"(accum_depth)   // %5
            :"0"(init_output),
            "1"(init_input),
            "2"(init_weight),
            "3"(init_bias),
            "4"(output_depth),
            "5"(accum_depth)
            :"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v24", "v25",
            "a2", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "s3",
            "fa0", "ft0"

        );

    }

    return CSINN_TRUE;
}

// best implementation from the software perspective
int csi_c906_fullyconnected_fp16_1(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct csi_tensor *weights,
                                 struct csi_tensor *bias,
                                 struct fc_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = (__fp16 *)weights->data;
    __fp16 *bias_data = (__fp16 *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = output->dim[0];
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    bool flag_bias = 1;     // default: fc layer include bias
    if (bias_data == NULL) {
        flag_bias = 0;
        bias_data = (__fp16 *)csi_mem_alloc(output_depth * 2);
    }

    for (int b = 0; b < batches; b++) {

        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;

        asm volatile(
            "vsetvli        zero, zero, e16, m1\n\t"    // set vl = 8

            "srai           t1, %5, 3\n\t"  // t1 = in_node >> 3 (k8)
            "andi           t2, %5, 7\n\t"  // t2 = in_node & 7
            "srai           t2, t2, 2\n\t"  // t2 = (in_node & 7) >> 2 (k4)
            "andi           t3, %5, 3\n\t"  // t3 = in_node & 3
            "srai           t3, t3, 1\n\t"  // t3 = (in_node & 3) >> 1 (k2)
            "andi           t4, %5, 1\n\t"  // t4 = in_node & 1 (k1)

            "srai           t0, %4, 3\n\t"  // out_node >> 3 (n8)
            "beqz           t0, 7f\n\t"     // jump to m1n4

        "1:\n\t"    // m1n8

            "vle.v          v24, (%3)\n\t"  // init out_tmp = bias_data
            "addi           %3, %3, 16\n\t"

                                            // %2 = weight_data addr
            "mv             t6, %1\n\t"     // t6 hold input 1 lines start addr

            "vle.v          v1, (%2)\n\t"   // pre-load weight-data
            "addi           %2, %2, 16\n\t"

            "flh            ft0, 0(t6)\n\t"     // pre-load input-data

            "beqz           t1, 3f\n\t"         // if k8 == 0, jump to subkernel_m1n8k4
            "mv             t5, t1\n\t"

            "2:\n\t"
            // m1n8k8
                "vle.v          v2, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            fa0, 2(t6)\n\t"
                "vfmacc.vf      v24, ft0, v1\n\t"   // 0


                "vle.v          v3, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            ft0, 4(t6)\n\t"
                "vfmacc.vf      v24, fa0, v2\n\t"   // 1


                "vle.v          v4, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            fa0, 6(t6)\n\t"
                "vfmacc.vf      v24, ft0, v3\n\t"   // 2


                "vle.v          v5, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            ft0, 8(t6)\n\t"
                "vfmacc.vf      v24, fa0, v4\n\t"   // 3


                "vle.v          v6, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            fa0, 10(t6)\n\t"
                "vfmacc.vf      v24, ft0, v5\n\t"   // 4


                "vle.v          v7, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            ft0, 12(t6)\n\t"
                "vfmacc.vf      v24, fa0, v6\n\t"   // 5


                "vle.v          v8, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            fa0, 14(t6)\n\t"
                "vfmacc.vf      v24, ft0, v7\n\t"   // 6
                "addi           t6, t6, 16\n\t"     // +8 elements, bump input_data to next k8 addr


                "vle.v          v1, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            ft0, 0(t6)\n\t"
                "vfmacc.vf      v24, fa0, v8\n\t"   // 7

                "addi           t5, t5, -1\n\t"
                "bnez           t5, 2b\n\t"

            "3:\n\t"
                "beqz           t2, 4f\n\t"     // if k4 == 0, jump to subkernel_m1n8k2
                // m1n8k4
                "vle.v          v2, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            fa0, 2(t6)\n\t"
                "vfmacc.vf      v24, ft0, v1\n\t"   // 0


                "vle.v          v3, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            ft0, 4(t6)\n\t"
                "vfmacc.vf      v24, fa0, v2\n\t"   // 1


                "vle.v          v4, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            fa0, 6(t6)\n\t"
                "vfmacc.vf      v24, ft0, v3\n\t"   // 2
                "addi           t6, t6, 8\n\t"      // +4 elements, bump pa to next k addr


                "vle.v          v1, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            ft0, 0(t6)\n\t"
                "vfmacc.vf      v24, fa0, v4\n\t"   // 3

            "4:\n\t"
                "beqz           t3, 5f\n\t"     // if k2 == 0, jump to subkernel_m1n8k1
                // m1n8k2
                "vle.v          v2, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            fa0, 2(t6)\n\t"
                "vfmacc.vf      v24, ft0, v1\n\t"   // 0
                "addi           t6, t6, 4\n\t"      // +2 elements, bump pa to next k addr


                "vle.v          v1, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            ft0, 0(t6)\n\t"
                "vfmacc.vf      v24, fa0, v2\n\t"   // 1


            "5:\n\t"
                "beqz           t4, 6f\n\t"     // if k1 == 0, jump to end kernel_m1n8
                // m1n8k1
                "vfmacc.vf      v24, ft0, v1\n\t"

                "addi           %2, %2, 16\n\t"     // ********************

        "6:\n\t"    // end m1n8

            // ********* bump pb to origin addr ************
            "addi           %2, %2, -16\n\t" // weight_data addr -= 8

            "vse.v          v24, (%0)\n\t"
            "addi           %0, %0, 16\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 1b\n\t"

        // "7:\n\t"
        "7:\n\t"    // m1n4

            // prepare for n4 n2 n1
            "andi           t2, %5, 7\n\t"  // t2 = k_tail
            "slli           t3, t2, 1\n\t"  // t3 = k_tail * 2

            "andi           t0, %4, 7\n\t"  // n & 7
            "srai           t0, t0, 2\n\t"  // (n & 7) >> 2
            "beqz           t0, 11f\n\t"    // jump to m1n2
            // start kernel_m1n4

            "vmv.v.x        v24, zero\n\t"
            "vmv.v.x        v25, zero\n\t"
            "vmv.v.x        v26, zero\n\t"
            "vmv.v.x        v27, zero\n\t"  // clear acc

            "flh            fs0, 0(%3)\n\t"
            "flh            fs1, 2(%3)\n\t"
            "flh            fs2, 4(%3)\n\t"
            "flh            fs3, 6(%3)\n\t"
            "addi           %3, %3, 8\n\t"

            "vfmv.s.f       v28, fs0\n\t"   // v28[0] = bias[0]
            "vfmv.s.f       v29, fs1\n\t"   // v29[0] = bias[1]
            "vfmv.s.f       v30, fs2\n\t"   // v30[0] = bias[2]
            "vfmv.s.f       v31, fs3\n\t"   // v31[0] = bias[3]

            // init addr for pa, pb and pc
            "slli           t0, %5, 1\n\t"  // t_tmp = k * 2

            "mv             t6, %1\n\t"     // t6 hold pa(input) 1 lines start addr

            "mv             a4, %2\n\t"
            "add            a5, a4, t0\n\t"
            "add            a6, a5, t0\n\t"
            "add            a7, a6, t0\n\t" // a4-a7 hold pb(weight) 4 cols addr

                                            // %0 hold pc(output) addr

            "mv             t5, t1\n\t"     // t5 = k8
            "beqz           t2, 9f\n\t"     // if k_tail == 0, jump to subkernel_m1n4k8

            "8:\n\t"
                // start subkernel_m1n4k_tail
                "vsetvli        zero, t2, e16, m1\n\t"
                "vle.v          v1, (t6)\n\t"
                "add            t6, t6, t3\n\t"
                "vle.v          v2, (a4)\n\t"
                "add            a4, a4, t3\n\t"
                "vle.v          v3, (a5)\n\t"
                "add            a5, a5, t3\n\t"
                "vle.v          v4, (a6)\n\t"
                "add            a6, a6, t3\n\t"
                "vle.v          v5, (a7)\n\t"
                "add            a7, a7, t3\n\t"
                "vfmacc.vv      v24, v1, v2\n\t"
                "vfmacc.vv      v25, v1, v3\n\t"
                "vfmacc.vv      v26, v1, v4\n\t"
                "vfmacc.vv      v27, v1, v5\n\t"

                "beqz           t1, 10f\n\t"    // if k8 == 0, jump to end kernel_m1n4
                "vsetvli        zero, zero, e16, m1\n\t"

            "9:\n\t"
                // start subkernel_m1n4k8
                "vle.v          v1, (t6)\n\t"
                "addi           t6, t6, 16\n\t"
                "vle.v          v2, (a4)\n\t"
                "addi           a4, a4, 16\n\t"
                "vle.v          v3, (a5)\n\t"
                "addi           a5, a5, 16\n\t"
                "vle.v          v4, (a6)\n\t"
                "addi           a6, a6, 16\n\t"
                "vle.v          v5, (a7)\n\t"
                "addi           a7, a7, 16\n\t"
                "vfmacc.vv      v24, v1, v2\n\t"
                "vfmacc.vv      v25, v1, v3\n\t"
                "vfmacc.vv      v26, v1, v4\n\t"
                "vfmacc.vv      v27, v1, v5\n\t"

                "addi           t5, t5, -1\n\t"
                "bnez           t5, 9b\n\t"


        "10:\n\t"   // end kernel_m1n4

            "vfredsum.vs    v28, v24, v28\n\t"  // v28[0] = v28[0](bias) + sum(v24[0..7])
            "vfredsum.vs    v29, v25, v29\n\t"
            "vfredsum.vs    v30, v26, v30\n\t"
            "vfredsum.vs    v31, v27, v31\n\t"
            "vfmv.f.s       fa0, v28\n\t"
            "vfmv.f.s       fa1, v29\n\t"
            "vfmv.f.s       fa2, v30\n\t"
            "vfmv.f.s       fa3, v31\n\t"
            "fsh            fa0, 0(%0)\n\t"
            "fsh            fa1, 2(%0)\n\t"
            "fsh            fa2, 4(%0)\n\t"
            "fsh            fa3, 6(%0)\n\t"

            "addi           %0, %0, 8\n\t"      // updata output start addr ( +4 cols)
            "slli           t0, %5, 3\n\t"      // t_tmp = k * 4 * 2
            "add            %2, %2, t0\n\t"     // updata pb start addr

        "11:\n\t"   // m1n2
            "andi           t0, %4, 3\n\t"  // n & 3
            "srai           t0, t0, 1\n\t"  // (n & 3) >> 1
            "beqz           t0, 15f\n\t"    // jump to m1n1
            // start kernel_m1n2

            "vmv.v.x        v24, zero\n\t"
            "vmv.v.x        v25, zero\n\t"  // clear acc

            "flh            fs0, 0(%3)\n\t"
            "flh            fs1, 2(%3)\n\t"
            "addi           %3, %3, 4\n\t"

            "vfmv.s.f       v28, fs0\n\t"   // v28[0] = bias[0]
            "vfmv.s.f       v29, fs1\n\t"   // v29[0] = bias[1]

            // init addr for pa, pb and pc
            "slli           t0, %5, 1\n\t"  // t_tmp = k * 2

            "mv             t6, %1\n\t"     // t6 hold pa(input) 1 lines start addr

            "mv             a4, %2\n\t"
            "add            a5, a4, t0\n\t" // a4-a5 hold pb(weight) 2 cols addr

                                            // %0 hold pc(output) addr

            "mv             t5, t1\n\t"     // t5 = k8
            "beqz           t2, 13f\n\t"    // if k_tail == 0, jump to subkernel_m1n2k8

            "12:\n\t"
                // start subkernel_m1n2k_tail
                "vsetvli        zero, t2, e16, m1\n\t"
                "vle.v          v1, (t6)\n\t"
                "add            t6, t6, t3\n\t"
                "vle.v          v2, (a4)\n\t"
                "add            a4, a4, t3\n\t"
                "vle.v          v3, (a5)\n\t"
                "add            a5, a5, t3\n\t"
                "vfmacc.vv      v24, v1, v2\n\t"
                "vfmacc.vv      v25, v1, v3\n\t"

                "beqz           t1, 14f\n\t"    // if k8 == 0, jump to end kernel_m1n2
                "vsetvli        zero, zero, e16, m1\n\t"

            "13:\n\t"
                // start subkernel_m1n2k8
                "vle.v          v1, (t6)\n\t"
                "addi           t6, t6, 16\n\t"
                "vle.v          v2, (a4)\n\t"
                "addi           a4, a4, 16\n\t"
                "vle.v          v3, (a5)\n\t"
                "addi           a5, a5, 16\n\t"
                "vfmacc.vv      v24, v1, v2\n\t"
                "vfmacc.vv      v25, v1, v3\n\t"

                "addi           t5, t5, -1\n\t"
                "bnez           t5, 13b\n\t"

        "14:\n\t"   // end kernel_m1n2

            "vfredsum.vs    v28, v24, v28\n\t"  // v28[0] = v28[0](bias) + sum(v24[0..7])
            "vfredsum.vs    v29, v25, v29\n\t"
            "vfmv.f.s       fa0, v28\n\t"
            "vfmv.f.s       fa1, v29\n\t"
            "fsh            fa0, 0(%0)\n\t"
            "fsh            fa1, 2(%0)\n\t"

            "addi           %0, %0, 4\n\t"      // updata output start addr ( +2 cols)
            "slli           t0, %5, 2\n\t"      // t_tmp = k * 2 * 2
            "add            %2, %2, t0\n\t"     // updata pb start addr

        "15:\n\t"   // m1n1
            "andi           t0, %4, 1\n\t"  // n & 1
            "beqz           t0, 19f\n\t"    // jump to ending
            // start kernel_m1n1
            "vmv.v.x        v24, zero\n\t"  // clear acc

            "flh            fs0, 0(%3)\n\t"
            "vfmv.s.f       v28, fs0\n\t"   // v28[0] = bias

            // init addr for pa, pb and pc
            "mv             t6, %1\n\t"     // t6 hold pa(input) 8 lines start addr

            "mv             a4, %2\n\t"     // a4 hold pb(weight) 1 cols addr

                                            // %0 hold pc(output) addr

            "mv             t5, t1\n\t"     // t5 = k8
            "beqz           t2, 17f\n\t"    // if k_tail == 0, jump to subkernel_m1n1k8

            "16:\n\t"
                // start subkernel_m1n1k_tail
                "vsetvli        zero, t2, e16, m1\n\t"
                "vle.v          v1, (t6)\n\t"
                "add            t6, t6, t3\n\t"
                "vle.v          v2, (a4)\n\t"
                "add            a4, a4, t3\n\t"
                "vfmacc.vv      v24, v1, v2\n\t"

                "beqz           t1, 18f\n\t"    // if k8 == 0, jump to end kernel_m1n1
                "vsetvli        zero, zero, e16, m1\n\t"

            "17:\n\t"
                // start subkernel_m1n1k8
                "vle.v          v1, (t6)\n\t"
                "addi           t6, t6, 16\n\t"
                "vle.v          v2, (a4)\n\t"
                "addi           a4, a4, 16\n\t"
                "vfmacc.vv      v24, v1, v2\n\t"

                "addi           t5, t5, -1\n\t"
                "bnez           t5, 17b\n\t"

        "18:\n\t"   // end kernel_m1n1
            "vfredsum.vs    v28, v24, v28\n\t"  // v28[0] = v28[0](bias) + sum(v24[0..7])
            "vfmv.f.s       fa0, v28\n\t"
            "fsh            fa0, 0(%0)\n\t"
            "addi           %0, %0, 2\n\t"

        "19:\n\t"   // ending

            :"=r"(init_output), // %0
            "=r"(init_input),   // %1
            "=r"(init_weight),  // %2
            "=r"(init_bias),    // %3
            "=r"(output_depth), // %4
            "=r"(accum_depth)   // %5
            :"0"(init_output),
            "1"(init_input),
            "2"(init_weight),
            "3"(init_bias),
            "4"(output_depth),
            "5"(accum_depth)
            :"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
            "a2", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
            "fa0", "fa1", "fa2", "fa3", "ft0", "fs0", "fs1", "fs2", "fs3"

        );
    }

    if (!flag_bias) {
        csi_mem_free(bias_data);
        bias_data = NULL;
    }

    return CSINN_TRUE;
}

// best performance measured on D1
int csi_c906_fullyconnected_fp16_2(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct csi_tensor *weights,
                                 struct csi_tensor *bias,
                                 struct fc_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = (__fp16 *)weights->data;
    __fp16 *bias_data = (__fp16 *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = output->dim[0];
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    bool flag_bias = 1;     // default: fc layer include bias
    if (bias_data == NULL) {
        flag_bias = 0;
        bias_data = (__fp16 *)csi_mem_alloc(output_depth * 2);
    }

    for (int b = 0; b < batches; b++) {

        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;

        asm volatile(
            "vsetvli        zero, zero, e16, m1\n\t"    // set vl = 8

            "srai           t4, %5, 3\n\t"  // k8
            "srai           t0, %4, 3\n\t"  // out_node >> 3 (n8)
            "beqz           t0, 3f\n\t"

        "1:\n\t"    // m1n8
            "vle.v          v4, (%3)\n\t"   // init out_tmp = bias_data
            "addi           %3, %3, 16\n\t"

            "mv             t1, %5\n\t"     // in_node (k)
            "mv             t6, %1\n\t"     // init input_data addr

            "2:\n\t"
                // m1n8k1
                "vle.v          v2, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "flh            fa0, 0(t6)\n\t"
                "vfmacc.vf      v4, fa0, v2\n\t"
                "addi           t6, t6, 2\n\t"

                "addi           t1, t1, -1\n\t"
                "bnez           t1, 2b\n\t"

            "vse.v          v4, (%0)\n\t"
            "addi           %0, %0, 16\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 1b\n\t"

        "3:\n\t"    // n_tail
            "andi           t0, %4, 7\n\t"  // n_tail
            "beqz           t0, 8f\n\t"    // if n_tail = 0, jump to ending

            // "mv             a2, %2\n\t"     // updata weight_data addr
            "andi           t2, %5, 7\n\t"  // k_tail
            "slli           t3, t2, 1\n\t"  // k_tail * 2

        "4:\n\t"
            "mv             t6, %1\n\t"     // init input_data addr

            "vmv.v.x        v4, zero\n\t"   // clear acc
            "flh            fa0, 0(%3)\n\t" // load bias
            "addi           %3, %3, 2\n\t"
            "vfmv.s.f       v3, fa0\n\t"    // v3[0] = bias

            "mv             t5, t4\n\t"     // t5 = k8
            "beqz           t2, 6f\n\t"

            "5:\n\t"
                // m1n1k_tail
                "vsetvli        zero, t2, e16, m1\n\t"
                "vle.v          v1, (t6)\n\t"
                "add            t6, t6, t3\n\t"
                "vle.v          v2, (%2)\n\t"
                "add            %2, %2, t3\n\t"
                "vfmacc.vv      v4, v1, v2\n\t"

                "beqz           t4, 7f\n\t"     // if k8 == 0, jump to end m1n1
                "vsetvli        zero, zero, e16, m1\n\t"

            "6:\n\t"
                // m1n1k8
                "vle.v          v1, (t6)\n\t"
                "addi           t6, t6, 16\n\t"
                "vle.v          v2, (%2)\n\t"
                "addi           %2, %2, 16\n\t"
                "vfmacc.vv      v4, v1, v2\n\t"

                "addi           t5, t5, -1\n\t"
                "bnez           t5, 6b\n\t"

        "7:\n\t"    // end m1n1
            "vfredsum.vs    v3, v4, v3\n\t"     // v3[0] = v3[0](bias) + sum(v4[0..7])
            "vfmv.f.s       fa0, v3\n\t"
            "fsh            fa0, 0(%0)\n\t"
            "addi           %0, %0, 2\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 4b\n\t"


        "8:\n\t"   // ending

            :"=r"(init_output), // %0
            "=r"(init_input),   // %1
            "=r"(init_weight),  // %2
            "=r"(init_bias),    // %3
            "=r"(output_depth), // %4
            "=r"(accum_depth)   // %5
            :"0"(init_output),
            "1"(init_input),
            "2"(init_weight),
            "3"(init_bias),
            "4"(output_depth),
            "5"(accum_depth)
            : "v1", "v2", "v3", "v4",
            "t0", "t1", "t2", "t3", "t4", "t5", "t6",
            "fa0"
        );

    }
    if (!flag_bias) {
        csi_mem_free(bias_data);
        bias_data = NULL;
    }

    return CSINN_TRUE;
}


int csi_c906_fullyconnected_init(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct csi_tensor *weights,
                                 struct csi_tensor *bias,
                                 struct fc_params *params)
{
    if (input->dtype == CSINN_DTYPE_FLOAT32) {
        params->base.bc = csi_c906_fullyconnected_f32;
    } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
        csi_c906_fc_gemv_transform_weight_fp16(weights);
        params->base.bc = csi_c906_fullyconnected_fp16_2;
        // params->base.bc = csi_c906_fullyconnected_fp16;
    }
    return CSINN_TRUE;
}
