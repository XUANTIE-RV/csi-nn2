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

#include "csi_c906.h"

static void reorder_matrixa_n8_fp16(__fp16 *src, __fp16 *dst, int row, int col)
{
    int vl = vsetvl_e16m1(8);
    int i = 0;
    for (; i + 7 < row; i += 8) {
        __fp16 *in_ptr = src + i * col;
        for (int j = 0; j < col; j++) {
            vfloat16m1_t _input = vlse16_v_f16m1(in_ptr, col * sizeof(__fp16), vl);
            in_ptr++;
            vse16_v_f16m1(dst, _input, vl);
            dst += vl;
        }
    }
    for (; i + 3 < row; i += 4) {
        __fp16 *in_ptr0 = src + i * col;
        __fp16 *in_ptr1 = in_ptr0 + col;
        __fp16 *in_ptr2 = in_ptr1 + col;
        __fp16 *in_ptr3 = in_ptr2 + col;

        int j = 0;
        for (; j + 7 < col; j += 8) {
            vfloat16m1_t _input0 = vle16_v_f16m1(in_ptr0, vl);
            in_ptr0 += vl;
            vfloat16m1_t _input1 = vle16_v_f16m1(in_ptr1, vl);
            in_ptr1 += vl;
            vfloat16m1_t _input2 = vle16_v_f16m1(in_ptr2, vl);
            in_ptr2 += vl;
            vfloat16m1_t _input3 = vle16_v_f16m1(in_ptr3, vl);
            in_ptr3 += vl;

            vsse16_v_f16m1(dst, 4 * sizeof(__fp16), _input0, vl);
            dst++;
            vsse16_v_f16m1(dst, 4 * sizeof(__fp16), _input1, vl);
            dst++;
            vsse16_v_f16m1(dst, 4 * sizeof(__fp16), _input2, vl);
            dst++;
            vsse16_v_f16m1(dst, 4 * sizeof(__fp16), _input3, vl);
            dst -= 3;
            dst += 32;
        }
        // col tail
        if (j < col) {
            int col_tail = col & 7;
            vl = vsetvl_e16m1(col_tail);
            vfloat16m1_t _input0 = vle16_v_f16m1(in_ptr0, vl);
            vfloat16m1_t _input1 = vle16_v_f16m1(in_ptr1, vl);
            vfloat16m1_t _input2 = vle16_v_f16m1(in_ptr2, vl);
            vfloat16m1_t _input3 = vle16_v_f16m1(in_ptr3, vl);

            vsse16_v_f16m1(dst, 4 * sizeof(__fp16), _input0, vl);
            dst++;
            vsse16_v_f16m1(dst, 4 * sizeof(__fp16), _input1, vl);
            dst++;
            vsse16_v_f16m1(dst, 4 * sizeof(__fp16), _input2, vl);
            dst++;
            vsse16_v_f16m1(dst, 4 * sizeof(__fp16), _input3, vl);
            dst -= 3;
            dst += 4 * col_tail;
        }
    }
    for (; i + 1 < row; i += 2) {
        __fp16 *in_ptr0 = src + i * col;
        __fp16 *in_ptr1 = in_ptr0 + col;
        vl = vsetvl_e16m1(8);
        int j = 0;
        for (; j + 7 < col; j += 8) {
            vfloat16m1_t _input0 = vle16_v_f16m1(in_ptr0, vl);
            in_ptr0 += vl;
            vfloat16m1_t _input1 = vle16_v_f16m1(in_ptr1, vl);
            in_ptr1 += vl;

            vsse16_v_f16m1(dst, 2 * sizeof(__fp16), _input0, vl);
            dst++;
            vsse16_v_f16m1(dst, 2 * sizeof(__fp16), _input1, vl);
            dst--;
            dst += 16;
        }
        // col tail
        if (j < col) {
            int col_tail = col & 7;
            vl = vsetvl_e16m1(col_tail);
            vfloat16m1_t _input0 = vle16_v_f16m1(in_ptr0, vl);
            vfloat16m1_t _input1 = vle16_v_f16m1(in_ptr1, vl);

            vsse16_v_f16m1(dst, 2 * sizeof(__fp16), _input0, vl);
            dst++;
            vsse16_v_f16m1(dst, 2 * sizeof(__fp16), _input1, vl);
            dst--;
            dst += 2 * col_tail;
        }
    }
    for (; i < row; i++) {
        __fp16 *in_ptr0 = src + i * col;
        vl = vsetvl_e16m1(8);
        int j = 0;
        for (; j + 7 < col; j += 8) {
            vfloat16m1_t _input0 = vle16_v_f16m1(in_ptr0, vl);
            in_ptr0 += vl;
            vse16_v_f16m1(dst, _input0, vl);
            dst += vl;
        }
        // col tail
        if (j < col) {
            int col_tail = col & 7;
            vl = vsetvl_e16m1(col_tail);
            vfloat16m1_t _input0 = vle16_v_f16m1(in_ptr0, vl);
            in_ptr0 += vl;
            vse16_v_f16m1(dst, _input0, vl);
            dst += vl;
        }
    }
}

static void reorder_matrixb_z8_fp16(__fp16 *src, __fp16 *dst, int row, int col)
{
    int vl = vsetvl_e16m1(8);
    int i = 0;
    for (; i + 7 < col; i += 8) {
        __fp16 *in_ptr = src + i;
        for (int j = 0; j < row; j++) {
            vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
            in_ptr += col;
            vse16_v_f16m1(dst, _input, vl);
            dst += vl;
        }
    }
    for (; i < col; i++) {
        __fp16 *in_ptr = src + i;
        vl = vsetvl_e16m1(8);
        int j = 0;
        for (; j + 7 < row; j += 8) {
            vfloat16m1_t _input0 = vlse16_v_f16m1(in_ptr, col * sizeof(__fp16), vl);
            in_ptr += 8 * col;
            vse16_v_f16m1(dst, _input0, vl);
            dst += vl;
        }
        // col tail
        if (j < row) {
            vl = vsetvl_e16m1(row & 7);
            vfloat16m1_t _input0 = vlse16_v_f16m1(in_ptr, col * sizeof(__fp16), vl);
            vse16_v_f16m1(dst, _input0, vl);
            dst += vl;
        }
    }
}

int csi_c906_matmul_fp32(struct csi_tensor *mat0, struct csi_tensor *mat1,
                         struct csi_tensor *output, struct matmul_params *params)
{
    return CSINN_TRUE;
}

int csi_c906_matmul_fp16(struct csi_tensor *mat0, struct csi_tensor *mat1,
                         struct csi_tensor *output, struct matmul_params *params)
{
    __fp16 *mat0_data = (__fp16 *)mat0->data;
    __fp16 *mat1_data = (__fp16 *)mat1->data;
    __fp16 *output_data = (__fp16 *)output->data;

    const int dims_count = mat0->dim_count;
    int batches = 1;

    /* compute the outer size */
    for (int i = 0; i < dims_count - 2; i++) {
        batches *= mat0->dim[i];
    }

    const int dim_m = mat0->dim[dims_count - (params->trans_a ? 1 : 2)];
    const int dim_k = mat0->dim[dims_count - (params->trans_a ? 2 : 1)];
    const int dim_n = mat1->dim[dims_count - (params->trans_b ? 2 : 1)];

    if (!params->trans_a && !params->trans_b) {
        __fp16 *in0 = (__fp16 *)csi_mem_alloc(dim_m * dim_k * sizeof(__fp16));
        __fp16 *in1 = (__fp16 *)csi_mem_alloc(dim_k * dim_n * sizeof(__fp16));

        for (int b = 0; b < batches; b++) {
            reorder_matrixa_n8_fp16(mat0_data, in0, dim_m, dim_k);
            reorder_matrixb_z8_fp16(mat1_data, in1, dim_k, dim_n);

            csi_c906_sgemm_kernel_fp16(output_data, in0, in1, dim_m, dim_k, dim_n, dim_n, NULL);

            mat0_data += dim_m * dim_k;
            mat1_data += dim_n * dim_k;
            output_data += dim_m * dim_n;
        }
        csi_mem_free(in0);
        csi_mem_free(in1);
    } else {
        csi_debug_error("Unsupport matrix transpose on C906\n");
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}