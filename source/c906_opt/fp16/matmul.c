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

#include "c906/c906.h"

/*************************************************************
  Matmul fp16_w_int8 performance on C906@1GHz
  -------------------------
  |      mkn     | GFlops |
  |-----------------------|
  |   49,32,49   |  2.29  |
  |   8,176,176  |  3.13  |
  |  8,1584,176  |  3.54  |
  | 384,512,512  |  3.71  |
  | 196,1536,384 |  3.83  |
  -------------------------

  Matmul fp16 performance on C906@1GHz
  ----------------------------------
  |              |      GFlops     |
  |      mkn     |-----------------|
  |              |  C906  |  RVV   |
  |--------------------------------|
  |   49,32,49   |  2.27  |  1.94  |
  |   8,176,176  |  1.2   |  1.59  |
  |  8,1584,176  |  0.36  |  1.81  |
  | 384,512,512  |  2.52  |  2.84  |
  | 196,1536,384 |  1.19  |  2.91  |
  -----------------------------------
 ************************************************************/

#define MATMUL_M_BLK 64
#define MATMUL_K_BLK 64
#define MATMUL_N_BLK 64

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

static int matmul_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                       struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    __fp16 *mat0_data = (__fp16 *)mat0->data;
    __fp16 *mat1_data = (__fp16 *)mat1->data;
    __fp16 *output_data = (__fp16 *)output->data;

    const int dims_count = mat0->dim_count;
    int batches_a = 1;
    int batches_b = 1;

    /* compute the outer size */
    for (int i = 0; i < dims_count - 2; i++) {
        batches_a *= mat0->dim[i];
    }
    for (int i = 0; i < mat1->dim_count - 2; i++) {
        batches_b *= mat1->dim[i];
    }

    const int dim_m = mat0->dim[dims_count - (params->trans_a ? 1 : 2)];
    const int dim_k = mat0->dim[dims_count - (params->trans_a ? 2 : 1)];
    const int dim_n = mat1->dim[mat1->dim_count - (params->trans_b ? 2 : 1)];

    if (!params->trans_a && !params->trans_b) {
        if (batches_a == batches_b) {
            __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
            __fp16 *in1 = (__fp16 *)shl_mem_alloc(dim_k * dim_n * sizeof(__fp16));

            for (int b = 0; b < batches_a; b++) {
                reorder_matrixa_n8_fp16(mat0_data, in0, dim_m, dim_k);
                reorder_matrixb_z8_fp16(mat1_data, in1, dim_k, dim_n);

                shl_c906_sgemm_kernel_fp16(output_data, in0, in1, dim_m, dim_k, dim_n, dim_n, NULL);

                mat0_data += dim_m * dim_k;
                mat1_data += dim_n * dim_k;
                output_data += dim_m * dim_n;
            }
            shl_mem_free(in0);
            shl_mem_free(in1);
            // requantize
            shl_rvv_sidcso_op_requantize_fp16(mat0, output, mat1);
        } else if (batches_a > 1 && batches_b == 1) {
            __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
            __fp16 *in1 = (__fp16 *)shl_mem_alloc(dim_k * dim_n * sizeof(__fp16));

            for (int b = 0; b < batches_a; b++) {
                reorder_matrixa_n8_fp16(mat0_data, in0, dim_m, dim_k);
                reorder_matrixb_z8_fp16(mat1_data, in1, dim_k, dim_n);

                shl_c906_sgemm_kernel_fp16(output_data, in0, in1, dim_m, dim_k, dim_n, dim_n, NULL);

                mat0_data += dim_m * dim_k;
                output_data += dim_m * dim_n;
            }
            shl_mem_free(in0);
            shl_mem_free(in1);
            // requantize
            shl_rvv_sidcso_op_requantize_fp16(mat0, output, mat1);
        } else {
            shl_debug_error("matmul unsupport this broadcast\n");
            return CSINN_FALSE;
        }
    } else {
        return shl_ref_matmul_quant(mat0, mat1, output, params);
    }

    return CSINN_TRUE;
}

/*
 * in0: original memory arrangement order
 * in1: Z32 Ztail
 */
static void shl_c906_matmul_4x32_fp16_w_int8(__fp16 *out, __fp16 *in0, int8_t *in1, int dim_m,
                                             int dim_k, int dim_n, int32_t zp, __fp16 scale)
{
    int m = 0;
    for (; m + 3 < dim_m; m += 4) {
        __fp16 *init_output0 = out + m * dim_n;
        __fp16 *init_output1 = init_output0 + dim_n;
        __fp16 *init_output2 = init_output1 + dim_n;
        __fp16 *init_output3 = init_output2 + dim_n;
        __fp16 *init_input0 = in0 + m * dim_k;
        __fp16 *init_input1 = init_input0 + dim_k;
        __fp16 *init_input2 = init_input1 + dim_k;
        __fp16 *init_input3 = init_input2 + dim_k;
        int8_t *init_weight = in1;

        int n = 0;
        while (n < dim_n) {
            int vl = vsetvl_e16m4(dim_n - n);
            // int8_t *init_weight = in1 + n;
            vfloat16m4_t _acc0 = vfmv_v_f_f16m4(0.0f, vl);
            vfloat16m4_t _acc1 = vfmv_v_f_f16m4(0.0f, vl);
            vfloat16m4_t _acc2 = vfmv_v_f_f16m4(0.0f, vl);
            vfloat16m4_t _acc3 = vfmv_v_f_f16m4(0.0f, vl);

            for (int k = 0; k < dim_k; k++) {
                vint8m2_t _weight = vle8_v_i8m2(init_weight, vl);
                vint16m4_t _weight_w = vwsub_vx_i16m4(_weight, zp, vl);
                vfloat16m4_t _weight_f = vfcvt_f_x_v_f16m4(_weight_w, vl);
                _weight_f = vfmul_vf_f16m4(_weight_f, scale, vl);
                _acc0 = vfmacc_vf_f16m4(_acc0, init_input0[k], _weight_f, vl);
                _acc1 = vfmacc_vf_f16m4(_acc1, init_input1[k], _weight_f, vl);
                _acc2 = vfmacc_vf_f16m4(_acc2, init_input2[k], _weight_f, vl);
                _acc3 = vfmacc_vf_f16m4(_acc3, init_input3[k], _weight_f, vl);
                init_weight += vl;
            }
            vse16_v_f16m4(init_output0, _acc0, vl);
            vse16_v_f16m4(init_output1, _acc1, vl);
            vse16_v_f16m4(init_output2, _acc2, vl);
            vse16_v_f16m4(init_output3, _acc3, vl);
            init_output0 += vl;
            init_output1 += vl;
            init_output2 += vl;
            init_output3 += vl;
            n += vl;
        }
    }
    for (; m + 1 < dim_m; m += 2) {
        __fp16 *init_output0 = out + m * dim_n;
        __fp16 *init_output1 = init_output0 + dim_n;
        __fp16 *init_input0 = in0 + m * dim_k;
        __fp16 *init_input1 = init_input0 + dim_k;
        int8_t *init_weight = in1;

        int n = 0;
        while (n < dim_n) {
            int vl = vsetvl_e16m4(dim_n - n);
            // int8_t *init_weight = in1 + n;
            vfloat16m4_t _acc0 = vfmv_v_f_f16m4(0.0f, vl);
            vfloat16m4_t _acc1 = vfmv_v_f_f16m4(0.0f, vl);

            for (int k = 0; k < dim_k; k++) {
                vint8m2_t _weight = vle8_v_i8m2(init_weight, vl);
                vint16m4_t _weight_w = vwsub_vx_i16m4(_weight, zp, vl);
                vfloat16m4_t _weight_f = vfcvt_f_x_v_f16m4(_weight_w, vl);
                _weight_f = vfmul_vf_f16m4(_weight_f, scale, vl);
                _acc0 = vfmacc_vf_f16m4(_acc0, init_input0[k], _weight_f, vl);
                _acc1 = vfmacc_vf_f16m4(_acc1, init_input1[k], _weight_f, vl);
                init_weight += vl;
            }
            vse16_v_f16m4(init_output0, _acc0, vl);
            vse16_v_f16m4(init_output1, _acc1, vl);
            init_output0 += vl;
            init_output1 += vl;
            n += vl;
        }
    }
    for (; m < dim_m; m++) {
        __fp16 *init_output0 = out + m * dim_n;
        __fp16 *init_input0 = in0 + m * dim_k;
        int8_t *init_weight = in1;

        int n = 0;
        while (n < dim_n) {
            int vl = vsetvl_e16m4(dim_n - n);
            // int8_t *init_weight = in1 + n;
            vfloat16m4_t _acc0 = vfmv_v_f_f16m4(0.0f, vl);

            for (int k = 0; k < dim_k; k++) {
                vint8m2_t _weight = vle8_v_i8m2(init_weight, vl);
                vint16m4_t _weight_w = vwsub_vx_i16m4(_weight, zp, vl);
                vfloat16m4_t _weight_f = vfcvt_f_x_v_f16m4(_weight_w, vl);
                _weight_f = vfmul_vf_f16m4(_weight_f, scale, vl);
                _acc0 = vfmacc_vf_f16m4(_acc0, init_input0[k], _weight_f, vl);
                init_weight += vl;
            }
            vse16_v_f16m4(init_output0, _acc0, vl);
            init_output0 += vl;
            n += vl;
        }
    }
}

static int matmul_fp16_w_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    __fp16 *mat0_data = (__fp16 *)mat0->data;
    int8_t *mat1_data = (int8_t *)mat1->data;
    __fp16 *output_data = (__fp16 *)output->data;

    const int dims_count = mat0->dim_count;
    int batches_a = 1;
    int batches_b = 1;

    /* compute the outer size */
    for (int i = 0; i < dims_count - 2; i++) {
        batches_a *= mat0->dim[i];
    }
    for (int i = 0; i < mat1->dim_count - 2; i++) {
        batches_b *= mat1->dim[i];
    }

    const int dim_m = mat0->dim[dims_count - (params->trans_a ? 1 : 2)];
    const int dim_k = mat0->dim[dims_count - (params->trans_a ? 2 : 1)];
    const int dim_n = mat1->dim[mat1->dim_count - (params->trans_b ? 2 : 1)];

    int32_t zp = mat1->qinfo->zero_point;
    float scale = mat1->qinfo->scale;

    if (!params->trans_a && !params->trans_b) {
        if (batches_a == batches_b) {
            for (int b = 0; b < batches_a; b++) {
                shl_c906_matmul_4x32_fp16_w_int8(output_data, mat0_data, mat1_data, dim_m, dim_k,
                                                 dim_n, zp, scale);

                mat0_data += dim_m * dim_k;
                mat1_data += dim_n * dim_k;
                output_data += dim_m * dim_n;
            }
        } else if (batches_a > 1 && batches_b == 1) {
            for (int b = 0; b < batches_a; b++) {
                /* TODO: mat1_data dequantize once */
                shl_c906_matmul_4x32_fp16_w_int8(output_data, mat0_data, mat1_data, dim_m, dim_k,
                                                 dim_n, zp, scale);
                mat0_data += dim_m * dim_k;
                output_data += dim_m * dim_n;
            }
        } else {
            shl_debug_error("matmul unsupport this broadcast\n");
            return CSINN_FALSE;
        }
    } else {
        return shl_ref_matmul_quant(mat0, mat1, output, params);
    }

    return CSINN_TRUE;
}

int shl_c906_matmul_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                         struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    const int dim_k = mat0->dim[mat0->dim_count - (params->trans_a ? 2 : 1)];
    if (mat1->is_const && mat1->dtype == CSINN_DTYPE_INT8) {
        return matmul_fp16_w_int8(mat0, mat1, output, params);
    } else if (mat1->dtype == CSINN_DTYPE_FLOAT16) {
        if (dim_k > MATMUL_K_BLK) {
            return shl_rvv_matmul_block_fp16(mat0, mat1, output, params, MATMUL_M_BLK, MATMUL_K_BLK,
                                             MATMUL_N_BLK);
        } else {
            return matmul_fp16(mat0, mat1, output, params);
        }
    } else {
        shl_debug_error("mat1 unsupport dtype: %d\n", mat1->dtype);
        return CSINN_FALSE;
    }
}

/* Z32 Ztail: adapt shl_c906_matmul_4x32_fp16_w_int8 */
static void shl_c906_matmul_reorder_weight_z32_int8(struct csinn_tensor *mat1)
{
    int8_t *mat1_data = (int8_t *)mat1->data;
    int dims_count = mat1->dim_count;
    int batch = 1;
    for (int i = 0; i < dims_count - 2; i++) {
        batch *= mat1->dim[i];
    }
    const int dim_k = mat1->dim[dims_count - 2];
    const int dim_n = mat1->dim[dims_count - 1];
    int8_t *mat_reorder = (int8_t *)shl_mem_alloc(dim_k * dim_n * sizeof(int8_t));

    for (int b = 0; b < batch; b++) {
        int8_t *init_mat = mat1_data + b * dim_k * dim_n;
        int8_t *dst = mat_reorder;

        int vl = vsetvl_e8m2(32);
        int n = 0;
        for (; n + 31 < dim_n; n += 32) {
            int8_t *in_ptr = init_mat + n;
            for (int k = 0; k < dim_k; k++) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, vl);
                in_ptr += dim_n;
                vse8_v_i8m2(dst, _input, vl);
                dst += vl;
            }
        }
        // Ztail
        if (n < dim_n) {
            int vl = vsetvl_e8m2(dim_n - n);
            int8_t *in_ptr = init_mat + n;
            for (int k = 0; k < dim_k; k++) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, vl);
                in_ptr += dim_n;
                vse8_v_i8m2(dst, _input, vl);
                dst += vl;
            }
        }
        memcpy(init_mat, mat_reorder, dim_k * dim_n * sizeof(int8_t));
    }
    shl_mem_free(mat_reorder);
}

int shl_c906_matmul_init_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    bool binary_model_op_init = shl_c906_get_binary_model_op_init(params->base.sess);
    struct csinn_callback *cb = params->base.cb;
    const int dim_k = mat1->dim[mat1->dim_count - (params->trans_b ? 1 : 2)];
    if (!params->trans_a && !params->trans_b) {
        if (mat0->dtype == CSINN_DTYPE_FLOAT16) {
            if (!binary_model_op_init) {
                if (mat1->is_const && mat1->dtype == CSINN_DTYPE_INT8) {
                    shl_c906_matmul_reorder_weight_z32_int8(mat1);
                } else if (mat1->dtype == CSINN_DTYPE_FLOAT16) {
                    if (dim_k > MATMUL_K_BLK) {
                        if (mat1->is_const) {
                            shl_rvv_matmul_reorder_weight_fp16(mat1, MATMUL_K_BLK, MATMUL_N_BLK);
                        }
                    }
                }
            }
            cb->exec = shl_c906_matmul_fp16;
        }
    }
    if (cb->exec == NULL) {
        shl_debug_warning(
            "matmul is not optimized to achieve under this condition, call reference func "
            "replaced.\n");
        cb->exec = shl_ref_matmul_quant;
    }
    return CSINN_TRUE;
}
