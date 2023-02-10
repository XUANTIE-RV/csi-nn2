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

#include "shl_c906.h"

/*
    change memory layout for weight matrix [out_nodes * in_nodes] by N(8) shape
*/
void shl_c906_reorder_weight_n8_fp16(__fp16 *src, __fp16 *dst, int m, int k, int ldx)
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
        shl_c906_memcpy(dst, src, sizeof(__fp16) * ldx);
        dst += k;
        src += k;
    }
}

void shl_c906_reorder_weight_n16_fp16(__fp16 *src, __fp16 *dst, int m, int k, int ldx)
{
    int i = 0;
    for (; i + 15 < m; i += 16) {
        for (int j = 0; j < k; j++) {
            dst[i * k + 16 * j + 0] = src[(i + 0) * k + j];
            dst[i * k + 16 * j + 1] = src[(i + 1) * k + j];
            dst[i * k + 16 * j + 2] = src[(i + 2) * k + j];
            dst[i * k + 16 * j + 3] = src[(i + 3) * k + j];
            dst[i * k + 16 * j + 4] = src[(i + 4) * k + j];
            dst[i * k + 16 * j + 5] = src[(i + 5) * k + j];
            dst[i * k + 16 * j + 6] = src[(i + 6) * k + j];
            dst[i * k + 16 * j + 7] = src[(i + 7) * k + j];
            dst[i * k + 16 * j + 8] = src[(i + 8) * k + j];
            dst[i * k + 16 * j + 9] = src[(i + 9) * k + j];
            dst[i * k + 16 * j + 10] = src[(i + 10) * k + j];
            dst[i * k + 16 * j + 11] = src[(i + 11) * k + j];
            dst[i * k + 16 * j + 12] = src[(i + 12) * k + j];
            dst[i * k + 16 * j + 13] = src[(i + 13) * k + j];
            dst[i * k + 16 * j + 14] = src[(i + 14) * k + j];
            dst[i * k + 16 * j + 15] = src[(i + 15) * k + j];
        }
    }
    // TODO: reorder N8
    dst += i * k;
    src += i * k;
    for (; i < m; i++) {
        shl_c906_memcpy(dst, src, sizeof(__fp16) * ldx);
        dst += k;
        src += k;
    }
}

void shl_c906_fc_gemv_transform_weight_fp16(struct csinn_tensor *weights)
{
    __fp16 *weight_data = (__fp16 *)weights->data;

    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes

    __fp16 *pa_reorder = (__fp16 *)shl_mem_alloc(n * k * sizeof(__fp16));
    shl_c906_reorder_weight_n16_fp16(weight_data, pa_reorder, n, k, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(__fp16));
    shl_mem_free(pa_reorder);
}

void shl_c906_reorder_weight_n16_fp16_w_int8(int8_t *src, int8_t *dst, int m, int k, int ldx)
{
    int i = 0;
    for (; i + 15 < m; i += 16) {
        for (int j = 0; j < k; j++) {
            dst[i * k + 16 * j + 0] = src[(i + 0) * k + j];
            dst[i * k + 16 * j + 1] = src[(i + 1) * k + j];
            dst[i * k + 16 * j + 2] = src[(i + 2) * k + j];
            dst[i * k + 16 * j + 3] = src[(i + 3) * k + j];
            dst[i * k + 16 * j + 4] = src[(i + 4) * k + j];
            dst[i * k + 16 * j + 5] = src[(i + 5) * k + j];
            dst[i * k + 16 * j + 6] = src[(i + 6) * k + j];
            dst[i * k + 16 * j + 7] = src[(i + 7) * k + j];
            dst[i * k + 16 * j + 8] = src[(i + 8) * k + j];
            dst[i * k + 16 * j + 9] = src[(i + 9) * k + j];
            dst[i * k + 16 * j + 10] = src[(i + 10) * k + j];
            dst[i * k + 16 * j + 11] = src[(i + 11) * k + j];
            dst[i * k + 16 * j + 12] = src[(i + 12) * k + j];
            dst[i * k + 16 * j + 13] = src[(i + 13) * k + j];
            dst[i * k + 16 * j + 14] = src[(i + 14) * k + j];
            dst[i * k + 16 * j + 15] = src[(i + 15) * k + j];
        }
    }
    // TODO: reorder N8
    dst += i * k;
    src += i * k;
    for (; i < m; i++) {
        shl_c906_memcpy(dst, src, sizeof(int8_t) * ldx);
        dst += k;
        src += k;
    }
}

void shl_c906_fc_gemv_transform_weight_fp16_w_int8(struct csinn_tensor *weights)
{
    int8_t *weight_data = (int8_t *)weights->data;

    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes

    int8_t *pa_reorder = (int8_t *)shl_mem_alloc(n * k * sizeof(int8_t));
    shl_c906_reorder_weight_n16_fp16_w_int8(weight_data, pa_reorder, n, k, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(int8_t));
    shl_mem_free(pa_reorder);
}

int shl_c906_fullyconnected_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *weights, struct csinn_tensor *bias,
                                 struct csinn_fc_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = (__fp16 *)weights->data;
    __fp16 *bias_data = (__fp16 *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    for (int b = 0; b < batches; b++) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;

        asm volatile(
            "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

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

            "0:\n\t"  // m1n8

            "vmv.v.x        v24, zero\n\t"  // clear
            "beqz           %3, 1f\n\t"     // if bias_data = NULL (bias->dim_count = 0)
            "vle.v          v24, (%3)\n\t"  // init out_tmp = bias_data
            "addi           %3, %3, 16\n\t"

            "1:\n\t"
            "mv             a2, %2\n\t"  // a2 = weight_data addr
            "mv             t6, %1\n\t"  // t6 hold input 1 lines start addr

            "vlse.v         v1, (a2), s2\n\t"  // pre-load weight-data
            "addi           a2, a2, 2\n\t"

            "flh            ft0, 0(t6)\n\t"  // pre-load input-data

            "beqz           t1, 3f\n\t"  // if k8 == 0, jump to subkernel_m1n8k4
            "mv             t5, t1\n\t"

            "2:\n\t"
            // m1n8k8
            "vlse.v         v2, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            fa0, 2(t6)\n\t"
            "vfmacc.vf      v24, ft0, v1\n\t"  // 0

            "vlse.v         v3, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            ft0, 4(t6)\n\t"
            "vfmacc.vf      v24, fa0, v2\n\t"  // 1

            "vlse.v         v4, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            fa0, 6(t6)\n\t"
            "vfmacc.vf      v24, ft0, v3\n\t"  // 2

            "vlse.v         v5, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            ft0, 8(t6)\n\t"
            "vfmacc.vf      v24, fa0, v4\n\t"  // 3

            "vlse.v         v6, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            fa0, 10(t6)\n\t"
            "vfmacc.vf      v24, ft0, v5\n\t"  // 4

            "vlse.v         v7, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            ft0, 12(t6)\n\t"
            "vfmacc.vf      v24, fa0, v6\n\t"  // 5

            "vlse.v         v8, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            fa0, 14(t6)\n\t"
            "vfmacc.vf      v24, ft0, v7\n\t"  // 6
            "addi           t6, t6, 16\n\t"    // +8 elements, bump input_data to next k8 addr

            "vlse.v         v1, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            ft0, 0(t6)\n\t"
            "vfmacc.vf      v24, fa0, v8\n\t"  // 7

            "addi           t5, t5, -1\n\t"
            "bnez           t5, 2b\n\t"

            "3:\n\t"
            "beqz           t2, 4f\n\t"  // if k4 == 0, jump to subkernel_m1n8k2
            // m1n8k4
            "vlse.v         v2, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            fa0, 2(t6)\n\t"
            "vfmacc.vf      v24, ft0, v1\n\t"  // 0

            "vlse.v         v3, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            ft0, 4(t6)\n\t"
            "vfmacc.vf      v24, fa0, v2\n\t"  // 1

            "vlse.v         v4, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            fa0, 6(t6)\n\t"
            "vfmacc.vf      v24, ft0, v3\n\t"  // 2
            "addi           t6, t6, 8\n\t"     // +4 elements, bump pa to next k addr

            "vlse.v         v1, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            ft0, 0(t6)\n\t"
            "vfmacc.vf      v24, fa0, v4\n\t"  // 3

            "4:\n\t"
            "beqz           t3, 5f\n\t"  // if k2 == 0, jump to subkernel_m1n8k1
            // m1n8k2
            "vlse.v         v2, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            fa0, 2(t6)\n\t"
            "vfmacc.vf      v24, ft0, v1\n\t"  // 0
            "addi           t6, t6, 4\n\t"     // +2 elements, bump pa to next k addr

            "vlse.v         v1, (a2), s2\n\t"
            "addi           a2, a2, 2\n\t"
            "flh            ft0, 0(t6)\n\t"
            "vfmacc.vf      v24, fa0, v2\n\t"  // 1

            "5:\n\t"
            "beqz           t4, 6f\n\t"  // if k1 == 0, jump to end kernel_m1n8
            // m1n8k1
            "vfmacc.vf      v24, ft0, v1\n\t"

            "6:\n\t"  // end m1n8

            "add            %2, %2, s3\n\t"  // weight_data addr + 8 lines

            "vse.v          v24, (%0)\n\t"
            "addi           %0, %0, 16\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 0b\n\t"

            "7:\n\t"                        // n_tail
            "andi           t0, %4, 7\n\t"  // n_tail
            "beqz           t0, 13f\n\t"    // if n_tail = 0, jump to ending

            "mv             a2, %2\n\t"     // updata weight_data addr
            "andi           t2, %5, 7\n\t"  // k_tail
            "slli           t3, t2, 1\n\t"  // k_tail * 2

            "8:\n\t"
            "mv             t6, %1\n\t"  // init input_data addr

            "vmv.v.x        v24, zero\n\t"   // clear acc
            "fmv.h.x        ft0, zero\n\t"   // clear
            "beqz           %3, 9f\n\t"      // if bias_data = NULL (bias->dim_count = 0)
            "flh            ft0, 0(%3)\n\t"  // else load bias_data
            "addi           %3, %3, 2\n\t"

            "9:\n\t"
            "vfmv.s.f       v25, ft0\n\t"  // v25[0] = bias

            "mv             t5, t1\n\t"  // t5 = k8
            "beqz           t2, 11f\n\t"

            "10:\n\t"
            // m1n1k_tail
            "vsetvli        zero, t2, e16, m1\n\t"
            "vle.v          v1, (t6)\n\t"
            "add            t6, t6, t3\n\t"
            "vle.v          v2, (a2)\n\t"
            "add            a2, a2, t3\n\t"
            "vfmacc.vv      v24, v1, v2\n\t"

            "beqz           t1, 12f\n\t"  // if k8 == 0, jump to end m1n1
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

            "12:\n\t"                           // end m1n1
            "vfredsum.vs    v25, v24, v25\n\t"  // v25[0] = v25[0](bias) + sum(v24[0..7])
            "vfmv.f.s       fa0, v25\n\t"
            "fsh            fa0, 0(%0)\n\t"
            "addi           %0, %0, 2\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 8b\n\t"

            "13:\n\t"

            : "=r"(init_output),   // %0
              "=r"(init_input),    // %1
              "=r"(init_weight),   // %2
              "=r"(init_bias),     // %3
              "=r"(output_depth),  // %4
              "=r"(accum_depth)    // %5
            : "0"(init_output), "1"(init_input), "2"(init_weight), "3"(init_bias),
              "4"(output_depth), "5"(accum_depth)
            : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v24", "v25", "a2", "t0", "t1", "t2",
              "t3", "t4", "t5", "t6", "s2", "s3", "fa0", "ft0"

        );
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, weights);
    return CSINN_TRUE;
}

/*
    best implementation from the software perspective
    loop unroll: k = 8
*/
int shl_c906_fullyconnected_pack8_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *weights, struct csinn_tensor *bias,
                                       struct csinn_fc_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = (__fp16 *)weights->data;
    __fp16 *bias_data = (__fp16 *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    bool flag_bias = 1;  // default: fc layer include bias
    if (bias_data == NULL) {
        flag_bias = 0;
        bias_data = (__fp16 *)shl_mem_alloc(output_depth * sizeof(__fp16));
    }

    for (int b = 0; b < batches; b++) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;

        asm volatile(
            "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

            "srai           t1, %5, 3\n\t"  // t1 = in_node >> 3 (k8)
            "andi           t2, %5, 7\n\t"  // t2 = in_node & 7
            "srai           t2, t2, 2\n\t"  // t2 = (in_node & 7) >> 2 (k4)
            "andi           t3, %5, 3\n\t"  // t3 = in_node & 3
            "srai           t3, t3, 1\n\t"  // t3 = (in_node & 3) >> 1 (k2)
            "andi           t4, %5, 1\n\t"  // t4 = in_node & 1 (k1)

            "srai           t0, %4, 3\n\t"  // out_node >> 3 (n8)
            "beqz           t0, 7f\n\t"     // jump to m1n4

            "1:\n\t"  // m1n8

            "vle.v          v24, (%3)\n\t"  // init out_tmp = bias_data
            "addi           %3, %3, 16\n\t"

            // %2 = weight_data addr
            "mv             t6, %1\n\t"  // t6 hold input 1 lines start addr

            "vle.v          v1, (%2)\n\t"  // pre-load weight-data
            "addi           %2, %2, 16\n\t"

            "flh            ft0, 0(t6)\n\t"  // pre-load input-data

            "beqz           t1, 3f\n\t"  // if k8 == 0, jump to subkernel_m1n8k4
            "mv             t5, t1\n\t"

            "2:\n\t"
            // m1n8k8
            "vle.v          v2, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            fa0, 2(t6)\n\t"
            "vfmacc.vf      v24, ft0, v1\n\t"  // 0

            "vle.v          v3, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            ft0, 4(t6)\n\t"
            "vfmacc.vf      v24, fa0, v2\n\t"  // 1

            "vle.v          v4, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            fa0, 6(t6)\n\t"
            "vfmacc.vf      v24, ft0, v3\n\t"  // 2

            "vle.v          v5, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            ft0, 8(t6)\n\t"
            "vfmacc.vf      v24, fa0, v4\n\t"  // 3

            "vle.v          v6, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            fa0, 10(t6)\n\t"
            "vfmacc.vf      v24, ft0, v5\n\t"  // 4

            "vle.v          v7, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            ft0, 12(t6)\n\t"
            "vfmacc.vf      v24, fa0, v6\n\t"  // 5

            "vle.v          v8, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            fa0, 14(t6)\n\t"
            "vfmacc.vf      v24, ft0, v7\n\t"  // 6
            "addi           t6, t6, 16\n\t"    // +8 elements, bump input_data to next k8 addr

            "vle.v          v1, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            ft0, 0(t6)\n\t"
            "vfmacc.vf      v24, fa0, v8\n\t"  // 7

            "addi           t5, t5, -1\n\t"
            "bnez           t5, 2b\n\t"

            "3:\n\t"
            "beqz           t2, 4f\n\t"  // if k4 == 0, jump to subkernel_m1n8k2
            // m1n8k4
            "vle.v          v2, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            fa0, 2(t6)\n\t"
            "vfmacc.vf      v24, ft0, v1\n\t"  // 0

            "vle.v          v3, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            ft0, 4(t6)\n\t"
            "vfmacc.vf      v24, fa0, v2\n\t"  // 1

            "vle.v          v4, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            fa0, 6(t6)\n\t"
            "vfmacc.vf      v24, ft0, v3\n\t"  // 2
            "addi           t6, t6, 8\n\t"     // +4 elements, bump pa to next k addr

            "vle.v          v1, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            ft0, 0(t6)\n\t"
            "vfmacc.vf      v24, fa0, v4\n\t"  // 3

            "4:\n\t"
            "beqz           t3, 5f\n\t"  // if k2 == 0, jump to subkernel_m1n8k1
            // m1n8k2
            "vle.v          v2, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            fa0, 2(t6)\n\t"
            "vfmacc.vf      v24, ft0, v1\n\t"  // 0
            "addi           t6, t6, 4\n\t"     // +2 elements, bump pa to next k addr

            "vle.v          v1, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "flh            ft0, 0(t6)\n\t"
            "vfmacc.vf      v24, fa0, v2\n\t"  // 1

            "5:\n\t"
            "beqz           t4, 6f\n\t"  // if k1 == 0, jump to end kernel_m1n8
            // m1n8k1
            "vfmacc.vf      v24, ft0, v1\n\t"

            "addi           %2, %2, 16\n\t"  // ********************

            "6:\n\t"  // end m1n8

            // ********* bump pb to origin addr ************
            "addi           %2, %2, -16\n\t"  // weight_data addr -= 8

            "vse.v          v24, (%0)\n\t"
            "addi           %0, %0, 16\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 1b\n\t"

            "7:\n\t"                        // m1n_tail
            "andi           t0, %4, 7\n\t"  // n_tail
            "beqz           t0, 12f\n\t"    // if n_tail = 0, jump to ending

            // prepare for n4 n2 n1
            "andi           t2, %5, 7\n\t"  // t2 = k_tail
            "slli           t3, t2, 1\n\t"  // t3 = k_tail * 2

            "8:\n\t"
            "mv             t6, %1\n\t"

            "vmv.v.x        v24, zero\n\t"
            "flh            fa0, 0(%3)\n\t"
            "addi           %3, %3, 2\n\t"
            "vfmv.s.f       v25, fa0\n\t"  // v25[0] = bias

            "mv             t5, t1\n\t"   // t5 = k8
            "beqz           t2, 10f\n\t"  // if k_tail = 0

            "9:\n\t"
            // m1n1k_tail
            "vsetvli        zero, t2, e16, m1\n\t"
            "vle.v          v1, (t6)\n\t"
            "add            t6, t6, t3\n\t"
            "vle.v          v2, (%2)\n\t"
            "add            %2, %2, t3\n\t"
            "vfmacc.vv      v24, v1, v2\n\t"

            "beqz           t1, 11f\n\t"  // if k8 == 0, jump to end m1n1
            "vsetvli        zero, zero, e16, m1\n\t"

            "10:\n\t"
            // m1n1k8
            "vle.v          v1, (t6)\n\t"
            "addi           t6, t6, 16\n\t"
            "vle.v          v2, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "vfmacc.vv      v24, v1, v2\n\t"

            "addi           t5, t5, -1\n\t"
            "bnez           t5, 10b\n\t"

            "11:\n\t"                           // end m1n1
            "vfredsum.vs    v25, v24, v25\n\t"  // v25[0] = v25[0](bias) + sum(v24[0..7])
            "vfmv.f.s       fa0, v25\n\t"
            "fsh            fa0, 0(%0)\n\t"
            "addi           %0, %0, 2\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 8b\n\t"

            "12:\n\t"  // ending

            : "=r"(init_output),   // %0
              "=r"(init_input),    // %1
              "=r"(init_weight),   // %2
              "=r"(init_bias),     // %3
              "=r"(output_depth),  // %4
              "=r"(accum_depth)    // %5
            : "0"(init_output), "1"(init_input), "2"(init_weight), "3"(init_bias),
              "4"(output_depth), "5"(accum_depth)
            : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v24", "v25", "t0", "t1", "t2", "t3",
              "t4", "t5", "t6", "fa0", "ft0"

        );
    }

    if (!flag_bias) {
        shl_mem_free(bias_data);
        bias_data = NULL;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, weights);
    return CSINN_TRUE;
}

/*
    loop unroll: k = 1
*/
int shl_c906_fullyconnected_pack8_fp16_1(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *weights, struct csinn_tensor *bias,
                                         struct csinn_fc_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = (__fp16 *)weights->data;
    __fp16 *bias_data = (__fp16 *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    bool flag_bias = 1;  // default: fc layer include bias
    if (bias_data == NULL) {
        flag_bias = 0;
        bias_data = (__fp16 *)shl_mem_alloc(output_depth * sizeof(__fp16));
    }

    for (int b = 0; b < batches; b++) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;

        asm volatile(
            "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

            "srai           t4, %5, 3\n\t"  // k8
            "srai           t0, %4, 3\n\t"  // out_node >> 3 (n8)
            "beqz           t0, 3f\n\t"

            "1:\n\t"                       // m1n8
            "vle.v          v4, (%3)\n\t"  // init out_tmp = bias_data
            "addi           %3, %3, 16\n\t"

            "mv             t1, %5\n\t"  // in_node (k)
            "mv             t6, %1\n\t"  // init input_data addr

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

            "3:\n\t"                        // n_tail
            "andi           t0, %4, 7\n\t"  // n_tail
            "beqz           t0, 8f\n\t"     // if n_tail = 0, jump to ending

            // "mv             a2, %2\n\t"     // updata weight_data addr
            "andi           t2, %5, 7\n\t"  // k_tail
            "slli           t3, t2, 1\n\t"  // k_tail * 2

            "4:\n\t"
            "mv             t6, %1\n\t"  // init input_data addr

            "vmv.v.x        v4, zero\n\t"    // clear acc
            "flh            fa0, 0(%3)\n\t"  // load bias
            "addi           %3, %3, 2\n\t"
            "vfmv.s.f       v3, fa0\n\t"  // v3[0] = bias

            "mv             t5, t4\n\t"  // t5 = k8
            "beqz           t2, 6f\n\t"

            "5:\n\t"
            // m1n1k_tail
            "vsetvli        zero, t2, e16, m1\n\t"
            "vle.v          v1, (t6)\n\t"
            "add            t6, t6, t3\n\t"
            "vle.v          v2, (%2)\n\t"
            "add            %2, %2, t3\n\t"
            "vfmacc.vv      v4, v1, v2\n\t"

            "beqz           t4, 7f\n\t"  // if k8 == 0, jump to end m1n1
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

            "7:\n\t"                         // end m1n1
            "vfredsum.vs    v3, v4, v3\n\t"  // v3[0] = v3[0](bias) + sum(v4[0..7])
            "vfmv.f.s       fa0, v3\n\t"
            "fsh            fa0, 0(%0)\n\t"
            "addi           %0, %0, 2\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 4b\n\t"

            "8:\n\t"  // ending

            : "=r"(init_output),   // %0
              "=r"(init_input),    // %1
              "=r"(init_weight),   // %2
              "=r"(init_bias),     // %3
              "=r"(output_depth),  // %4
              "=r"(accum_depth)    // %5
            : "0"(init_output), "1"(init_input), "2"(init_weight), "3"(init_bias),
              "4"(output_depth), "5"(accum_depth)
            : "v1", "v2", "v3", "v4", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "fa0");
    }
    if (!flag_bias) {
        shl_mem_free(bias_data);
        bias_data = NULL;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, weights);
    return CSINN_TRUE;
}

/*
    best performance measured on D1
    loop unroll: k = 1 && pack16
*/
int shl_c906_fullyconnected_pack16_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *weights, struct csinn_tensor *bias,
                                        struct csinn_fc_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = NULL;
    __fp16 *bias_data = (__fp16 *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    bool flag_bias = 1;  // default: fc layer include bias
    if (bias_data == NULL) {
        flag_bias = 0;
        bias_data = (__fp16 *)shl_mem_alloc(output_depth * sizeof(__fp16));
    }

    __fp16 *weights_fp16 = NULL;
    if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
        // TODO: support per-channel quantization
        int32_t zp = weights->qinfo->zero_point;
        float scale = weights->qinfo->scale;
        int size = csinn_tensor_size(weights);
        int8_t *weights_int8 = (int8_t *)weights->data;
        weights_fp16 = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
        shl_rvv_dequantize_i8_to_f16(weights_int8, weights_fp16, size, zp, scale);
        weights_data = weights_fp16;
    } else if (weights->dtype == CSINN_DTYPE_FLOAT16) {
        weights_data = (__fp16 *)weights->data;
    } else {
        shl_debug_error("weights unsupport dtype: %d\n", weights->dtype);
        return CSINN_FALSE;
    }

    for (int b = 0; b < batches; b++) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;

        asm volatile(
            "vsetvli        zero, zero, e16, m2\n\t"  // set vl = 16

            "srai           t4, %5, 4\n\t"  // k16
            "srai           t0, %4, 4\n\t"  // out_node >> 4 (n16)
            "beqz           t0, 3f\n\t"

            "1:\n\t"                       // m1n8
            "vle.v          v4, (%3)\n\t"  // init out_tmp = bias_data
            "addi           %3, %3, 32\n\t"

            "mv             t1, %5\n\t"  // in_node (k)
            "mv             t6, %1\n\t"  // init input_data addr

            "2:\n\t"
            // m1n8k1
            "vle.v          v2, (%2)\n\t"
            "addi           %2, %2, 32\n\t"
            "flh            fa0, 0(t6)\n\t"
            "vfmacc.vf      v4, fa0, v2\n\t"
            "addi           t6, t6, 2\n\t"

            "addi           t1, t1, -1\n\t"
            "bnez           t1, 2b\n\t"

            "vse.v          v4, (%0)\n\t"
            "addi           %0, %0, 32\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 1b\n\t"

            "3:\n\t"                         // n_tail
            "andi           t0, %4, 15\n\t"  // n_tail
            "beqz           t0, 8f\n\t"      // if n_tail = 0, jump to ending

            // "mv             a2, %2\n\t"     // updata weight_data addr
            "andi           t2, %5, 15\n\t"  // k_tail
            "slli           t3, t2, 1\n\t"   // k_tail * 2

            "4:\n\t"
            "mv             t6, %1\n\t"  // init input_data addr

            "vmv.v.x        v4, zero\n\t"    // clear acc
            "flh            fa0, 0(%3)\n\t"  // load bias
            "addi           %3, %3, 2\n\t"
            "vfmv.s.f       v6, fa0\n\t"  // v6[0] = bias

            "mv             t5, t4\n\t"  // t5 = k8
            "beqz           t2, 6f\n\t"

            "5:\n\t"
            // m1n1k_tail
            "vsetvli        zero, t2, e16, m2\n\t"
            "vle.v          v0, (t6)\n\t"
            "add            t6, t6, t3\n\t"
            "vle.v          v2, (%2)\n\t"
            "add            %2, %2, t3\n\t"
            "vfmacc.vv      v4, v0, v2\n\t"

            "beqz           t4, 7f\n\t"  // if k8 == 0, jump to end m1n1
            "vsetvli        zero, zero, e16, m2\n\t"

            "6:\n\t"
            // m1n1k8
            "vle.v          v0, (t6)\n\t"
            "addi           t6, t6, 32\n\t"
            "vle.v          v2, (%2)\n\t"
            "addi           %2, %2, 32\n\t"
            "vfmacc.vv      v4, v0, v2\n\t"

            "addi           t5, t5, -1\n\t"
            "bnez           t5, 6b\n\t"

            "7:\n\t"                         // end m1n1
            "vfredsum.vs    v6, v4, v6\n\t"  // v6[0] = v6[0](bias) + sum(v4[0..7])
            "vfmv.f.s       fa0, v6\n\t"
            "fsh            fa0, 0(%0)\n\t"
            "addi           %0, %0, 2\n\t"

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 4b\n\t"

            "8:\n\t"  // ending

            : "=r"(init_output),   // %0
              "=r"(init_input),    // %1
              "=r"(init_weight),   // %2
              "=r"(init_bias),     // %3
              "=r"(output_depth),  // %4
              "=r"(accum_depth)    // %5
            : "0"(init_output), "1"(init_input), "2"(init_weight), "3"(init_bias),
              "4"(output_depth), "5"(accum_depth)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "t0", "t1", "t2", "t3", "t4", "t5",
              "t6", "fa0");
    }
    if (!flag_bias) {
        shl_mem_free(bias_data);
        bias_data = NULL;
    }
    if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
        shl_mem_free(weights_fp16);
        return CSINN_TRUE;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, weights);
    return CSINN_TRUE;
}

int shl_c906_fullyconnected_pack16_output16_fp16(struct csinn_tensor *input,
                                                 struct csinn_tensor *output,
                                                 struct csinn_tensor *weights,
                                                 struct csinn_tensor *bias,
                                                 struct csinn_fc_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = NULL;
    __fp16 *bias_data = (__fp16 *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    __fp16 *weights_fp16 = NULL;
    if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
        int32_t zp = weights->qinfo->zero_point;
        float scale = weights->qinfo->scale;
        int size = csinn_tensor_size(weights);
        int8_t *weights_int8 = (int8_t *)weights->data;
        weights_fp16 = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
        shl_rvv_dequantize_i8_to_f16(weights_int8, weights_fp16, size, zp, scale);
        weights_data = weights_fp16;
    } else if (weights->dtype == CSINN_DTYPE_FLOAT16) {
        weights_data = (__fp16 *)weights->data;
    } else {
        shl_debug_error("weights unsupport dtype: %d\n", weights->dtype);
        return CSINN_FALSE;
    }

    int packn = 16;
    int vl = 16;
    int b = 0;
    for (; b + 7 < batches; b += 8) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_output2 = init_output + output_depth;
        __fp16 *init_output3 = init_output2 + output_depth;
        __fp16 *init_output4 = init_output3 + output_depth;
        __fp16 *init_output5 = init_output4 + output_depth;
        __fp16 *init_output6 = init_output5 + output_depth;
        __fp16 *init_output7 = init_output6 + output_depth;
        __fp16 *init_output8 = init_output7 + output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_input2 = init_input + accum_depth;
        __fp16 *init_input3 = init_input2 + accum_depth;
        __fp16 *init_input4 = init_input3 + accum_depth;
        __fp16 *init_input5 = init_input4 + accum_depth;
        __fp16 *init_input6 = init_input5 + accum_depth;
        __fp16 *init_input7 = init_input6 + accum_depth;
        __fp16 *init_input8 = init_input7 + accum_depth;
        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;
        int n = output_depth;
        while (n > 0) {
            __fp16 *in_ptr = init_input;
            __fp16 *in_ptr2 = init_input2;
            __fp16 *in_ptr3 = init_input3;
            __fp16 *in_ptr4 = init_input4;
            __fp16 *in_ptr5 = init_input5;
            __fp16 *in_ptr6 = init_input6;
            __fp16 *in_ptr7 = init_input7;
            __fp16 *in_ptr8 = init_input8;
            vfloat16m2_t _acc = vle16_v_f16m2(init_bias, vl);
            vfloat16m2_t _acc2 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc3 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc4 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc5 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc6 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc7 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc8 = vmv_v_v_f16m2(_acc, vl);

            init_bias += vl;
            int k = accum_depth;
            while (k > 0) {
                vfloat16m2_t _weight = vle16_v_f16m2(init_weight, vl);
                _acc = vfmacc_vf_f16m2(_acc, *in_ptr, _weight, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, *in_ptr2, _weight, vl);
                _acc3 = vfmacc_vf_f16m2(_acc3, *in_ptr3, _weight, vl);
                _acc4 = vfmacc_vf_f16m2(_acc4, *in_ptr4, _weight, vl);
                _acc5 = vfmacc_vf_f16m2(_acc5, *in_ptr5, _weight, vl);
                _acc6 = vfmacc_vf_f16m2(_acc6, *in_ptr6, _weight, vl);
                _acc7 = vfmacc_vf_f16m2(_acc7, *in_ptr7, _weight, vl);
                _acc8 = vfmacc_vf_f16m2(_acc8, *in_ptr8, _weight, vl);
                init_weight += vl;
                in_ptr++;
                in_ptr2++;
                in_ptr3++;
                in_ptr4++;
                in_ptr5++;
                in_ptr6++;
                in_ptr7++;
                in_ptr8++;
                k--;
            }
            vse16_v_f16m2(init_output, _acc, vl);
            vse16_v_f16m2(init_output2, _acc2, vl);
            vse16_v_f16m2(init_output3, _acc3, vl);
            vse16_v_f16m2(init_output4, _acc4, vl);
            vse16_v_f16m2(init_output5, _acc5, vl);
            vse16_v_f16m2(init_output6, _acc6, vl);
            vse16_v_f16m2(init_output7, _acc7, vl);
            vse16_v_f16m2(init_output8, _acc8, vl);
            init_output += vl;
            init_output2 += vl;
            init_output3 += vl;
            init_output4 += vl;
            init_output5 += vl;
            init_output6 += vl;
            init_output7 += vl;
            init_output8 += vl;
            n -= vl;
        }
    }
    for (; b + 3 < batches; b += 4) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_output2 = init_output + output_depth;
        __fp16 *init_output3 = init_output2 + output_depth;
        __fp16 *init_output4 = init_output3 + output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_input2 = init_input + accum_depth;
        __fp16 *init_input3 = init_input2 + accum_depth;
        __fp16 *init_input4 = init_input3 + accum_depth;

        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;
        int n = output_depth;
        while (n > 0) {
            __fp16 *in_ptr = init_input;
            __fp16 *in_ptr2 = init_input2;
            __fp16 *in_ptr3 = init_input3;
            __fp16 *in_ptr4 = init_input4;

            vfloat16m2_t _acc = vle16_v_f16m2(init_bias, vl);
            vfloat16m2_t _acc2 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc3 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc4 = vmv_v_v_f16m2(_acc, vl);

            init_bias += vl;
            int k = accum_depth;
            while (k > 0) {
                vfloat16m2_t _weight = vle16_v_f16m2(init_weight, vl);
                _acc = vfmacc_vf_f16m2(_acc, *in_ptr, _weight, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, *in_ptr2, _weight, vl);
                _acc3 = vfmacc_vf_f16m2(_acc3, *in_ptr3, _weight, vl);
                _acc4 = vfmacc_vf_f16m2(_acc4, *in_ptr4, _weight, vl);
                init_weight += vl;
                in_ptr++;
                in_ptr2++;
                in_ptr3++;
                in_ptr4++;
                k--;
            }
            vse16_v_f16m2(init_output, _acc, vl);
            vse16_v_f16m2(init_output2, _acc2, vl);
            vse16_v_f16m2(init_output3, _acc3, vl);
            vse16_v_f16m2(init_output4, _acc4, vl);
            init_output += vl;
            init_output2 += vl;
            init_output3 += vl;
            init_output4 += vl;
            n -= vl;
        }
    }
    for (; b + 1 < batches; b += 2) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_output2 = init_output + output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_input2 = init_input + accum_depth;

        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;
        int n = output_depth;
        while (n > 0) {
            __fp16 *in_ptr = init_input;
            __fp16 *in_ptr2 = init_input2;
            vfloat16m2_t _acc = vle16_v_f16m2(init_bias, vl);
            vfloat16m2_t _acc2 = vmv_v_v_f16m2(_acc, vl);
            init_bias += vl;
            int k = accum_depth;
            while (k > 0) {
                vfloat16m2_t _weight = vle16_v_f16m2(init_weight, vl);
                _acc = vfmacc_vf_f16m2(_acc, *in_ptr, _weight, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, *in_ptr2, _weight, vl);
                init_weight += vl;
                in_ptr++;
                in_ptr2++;
                k--;
            }
            vse16_v_f16m2(init_output, _acc, vl);
            vse16_v_f16m2(init_output2, _acc2, vl);
            init_output += vl;
            init_output2 += vl;
            n -= vl;
        }
    }
    for (; b < batches; b++) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_input = input_data + b * accum_depth;

        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;
        int n = output_depth;
        while (n > 0) {
            __fp16 *in_ptr = init_input;
            vfloat16m2_t _acc = vle16_v_f16m2(init_bias, vl);
            init_bias += vl;
            int k = accum_depth;
            while (k > 0) {
                vfloat16m2_t _weight = vle16_v_f16m2(init_weight, vl);
                _acc = vfmacc_vf_f16m2(_acc, *in_ptr, _weight, vl);
                init_weight += vl;
                in_ptr++;
                k--;
            }
            vse16_v_f16m2(init_output, _acc, vl);
            init_output += vl;
            n -= vl;
        }
    }
    if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
        shl_mem_free(weights_fp16);
        return CSINN_TRUE;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, weights);
    return CSINN_TRUE;
}

int shl_c906_fullyconnected_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *weights, struct csinn_tensor *bias,
                                      struct csinn_fc_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    if (weights->mtype != CSINN_MEM_TYPE_CPU_ALIGNED) {
        if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
            shl_c906_fc_gemv_transform_weight_fp16_w_int8(weights);
        } else if (weights->dtype == CSINN_DTYPE_FLOAT16) {
            shl_c906_fc_gemv_transform_weight_fp16(weights);
        } else {
            shl_debug_error("weights unsupport dtype: %d\n", weights->dtype);
            return CSINN_FALSE;
        }
    }
    int output_depth = weights->dim[weights->dim_count - 2];
    if (bias != NULL && output_depth % 16 == 0) {
        cb->exec = shl_c906_fullyconnected_pack16_output16_fp16;
    } else {
        cb->exec = shl_c906_fullyconnected_pack16_fp16;
    }

    return CSINN_TRUE;
}
