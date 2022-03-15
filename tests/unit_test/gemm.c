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

/* CSI-NN2 version 1.13.x */

#include "./valid_data/gemm.dat"
#include "csi_nn.h"
#include "csi_thead_rvv.h"
#include "math_snr.h"
#include "test_utils.h"

void verify_gemm_reorderA(void *ma_data, void *ref_ma_data, void (*reorder)(), int m, int k,
                          int ldx, enum csinn_dtype_enum dtype)
{
    void *out_data = csi_mem_alloc(m * k * sizeof(float));
    reorder(ma_data, out_data, m, k, ldx);
    evaluate_error(out_data, ref_ma_data, m * k, dtype);
    csi_mem_free(out_data);
}

void verify_gemm_reorderB(void *mb_data, void *ref_mb_data, void (*reorder)(), int k, int n,
                          int ldx, enum csinn_dtype_enum dtype)
{
    void *out_data = csi_mem_alloc(k * n * sizeof(float));
    reorder(mb_data, out_data, k, n, ldx);
    evaluate_error(out_data, ref_mb_data, k * n, dtype);
    csi_mem_free(out_data);
}

void verify_gemm_compute(void *ma_data, void *mb_data, void *bias_data, void *ref_data,
                         void (*compute)(), int m, int k, int n, int ldx,
                         enum csinn_dtype_enum dtype)
{
    void *out_data = csi_mem_alloc(m * n * sizeof(float));
    compute(out_data, ma_data, mb_data, m, k, n, ldx, bias_data);
    evaluate_error(out_data, ref_data, m * n, dtype);
    csi_mem_free(out_data);
}

int main(int argc, char **argv)
{
    init_testsuite("Test function of gemm for RVV.\n");

    verify_gemm_reorderA(gemm_fp32_a, gemm_fp32_a1, csi_nn_rvv_reorder_kernel_n8_fp32, 31, 16, 16,
                         CSINN_DTYPE_FLOAT32);
    verify_gemm_reorderB(gemm_fp32_b, gemm_fp32_b1, csi_nn_rvv_reorder_input_z8_fp32, 16, 20, 20,
                         CSINN_DTYPE_FLOAT32);
    verify_gemm_compute(gemm_fp32_a1, gemm_fp32_b1, gemm_fp32_bias, gemm_fp32_c,
                        csi_nn_rvv_gemm_8x8_fp32, 31, 16, 20, 20, CSINN_DTYPE_FLOAT32);

    verify_gemm_reorderA(gemm_fp16_a, gemm_fp16_a1, csi_nn_rvv_reorder_kernel_n8_fp16, 31, 16, 16,
                         CSINN_DTYPE_FLOAT16);
    verify_gemm_reorderB(gemm_fp16_b, gemm_fp16_b1, csi_nn_rvv_reorder_input_z16_fp16, 16, 20, 20,
                         CSINN_DTYPE_FLOAT16);
    verify_gemm_compute(gemm_fp16_a1, gemm_fp16_b1, gemm_fp16_bias, gemm_fp16_c,
                        csi_nn_rvv_gemm_8x16_fp16, 31, 16, 20, 20, CSINN_DTYPE_FLOAT16);

    return done_testing();
}
