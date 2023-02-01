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

/* SHL version 2.1.x */

#include "shl_thead_rvm.h"

static inline void wg_bxf3s1_reorder_kernel_nhwc_fp16(__fp16 *dst, const __fp16 *src, int N, int K,
                                                      int mcols)
{
    int m2rows = mcols;
    int n = 0;
    for (; n + m2rows - 1 < N; n += m2rows) {
        const __fp16 *src_n = src + n * K;
        int k = 0;
        for (; k + mcols - 1 < K; k += mcols) {
            const __fp16 *src_k = src_n + k;
            for (int i = 0; i < m2rows; i++) {
                const __fp16 *src_i = src_k + i * K;
                memcpy(dst, src_i, mcols * sizeof(__fp16));
                dst += mcols;
            }
        }
        // k_tail
        if (k < K) {
            const __fp16 *src_k = src_n + k;
            for (int i = 0; i < m2rows; i++) {
                const __fp16 *src_i = src_k + i * K;
                memcpy(dst, src_i, (K - k) * sizeof(__fp16));
                dst += mcols;
            }
        }
    }
    // n_tail
    if (n < N) {
        const __fp16 *src_n = src + n * K;
        int k = 0;
        for (; k + mcols - 1 < K; k += mcols) {
            const __fp16 *src_k = src_n + k;
            for (int i = 0; i < m2rows; i++) {
                if (i < N - n) {
                    const __fp16 *src_i = src_k + i * K;
                    memcpy(dst, src_i, mcols * sizeof(__fp16));
                }
                dst += mcols;
            }
        }
        // k_tail
        if (k < K) {
            const __fp16 *src_k = src_n + k;
            for (int i = 0; i < m2rows; i++) {
                if (i < N - n) {
                    const __fp16 *src_i = src_k + i * K;
                    memcpy(dst, src_i, (K - k) * sizeof(__fp16));
                }
                dst += mcols;
            }
        }
    }
}

/******************************************************************************************
 * kernel layout before:  [O, 3, 3, I]
 * kernel layout after :  [O, 6, 6, I] --> [36, O, I]
 ******************************************************************************************/
void shl_rvm_wg_b4f3s1_trans_kernel_nhwc_fp16(struct csinn_tensor *src_kernel,
                                              struct csinn_tensor *dst_kernel)
{
    int32_t out_c = src_kernel->dim[0];
    int32_t in_c = src_kernel->dim[3];
    __fp16 *kernel_data = (__fp16 *)src_kernel->data;
    /* kernel transform matrix: G
    G =
        ⎡1/4     0     0  ⎤
        ⎢-1/6  -1/6   -1/6⎥
        ⎢-1/6   1/6   -1/6⎥
        ⎢1/24  1/12   1/6 ⎥
        ⎢1/24  -1/12  1/6 ⎥
        ⎣ 0      0     1  ⎦
    */
    const __fp16 G[6][3] = {{1.0f / 4, 0.0f, 0.0f},
                            {-1.0f / 6, -1.0f / 6, -1.0f / 6},
                            {-1.0f / 6, 1.0f / 6, -1.0f / 6},
                            {1.0f / 24, 1.0f / 12, 1.0f / 6},
                            {1.0f / 24, -1.0f / 12, 1.0f / 6},
                            {0.0f, 0.0f, 1.0f}};

    csinn_tensor_copy(dst_kernel, src_kernel);

    // for kernel transform buf, 3x3 --> 6x6
    __fp16 *kernel_tm_1 = (__fp16 *)shl_mem_alloc(out_c * 6 * 3 * in_c * sizeof(__fp16));
    __fp16 *kernel_tm_2 = (__fp16 *)shl_mem_alloc(out_c * 6 * 6 * in_c * sizeof(__fp16));

    const int vl_16 = vsetvl_e16m1(in_c);
    int vl = 0;
    for (int oc = 0; oc < out_c; oc++) {
        // G * g
        __fp16 *kernel_tm_10 = kernel_tm_1 + oc * 6 * 3 * in_c;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 3; j++) {
                __fp16 *res_ptr = kernel_tm_10 + (i * 3 + j) * in_c;
                for (int k = 0; k < 3; k++) {
                    const __fp16 *k_ptr = kernel_data + ((oc * 3 + k) * 3 + j) * in_c;
                    const __fp16 g0 = G[i][k];
                    vfloat16m1_t _g0 = vfmv_v_f_f16m1(g0, vl_16);
                    int c = 0;
                    for (; c + vl_16 - 1 < in_c; c += vl_16) {
                        const __fp16 *k0 = k_ptr + c;
                        __fp16 *res0 = res_ptr + c;
                        vfloat16m1_t _res = vle16_v_f16m1(res0, vl_16);
                        vfloat16m1_t _kernel = vle16_v_f16m1(k0, vl_16);
                        _res = vfmacc_vv_f16m1(_res, _g0, _kernel, vl_16);
                        vse16_v_f16m1(res0, _res, vl_16);
                    }
                    while (c < in_c) {
                        const __fp16 *k0 = k_ptr + c;
                        __fp16 *res0 = res_ptr + c;
                        vl = vsetvl_e16m1(in_c - c);
                        vfloat16m1_t _res = vle16_v_f16m1(res0, vl);
                        vfloat16m1_t _kernel = vle16_v_f16m1(k0, vl);
                        _res = vfmacc_vf_f16m1(_res, g0, _kernel, vl);
                        vse16_v_f16m1(res0, _res, vl);
                        c += vl;
                    }
                }
            }
        }

        // * GT
        __fp16 *kernel_tm_20 = kernel_tm_2 + oc * 36 * in_c;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                __fp16 *res_ptr = kernel_tm_20 + (i * 6 + j) * in_c;
                for (int k = 0; k < 3; k++) {
                    const __fp16 *tmp_ptr = kernel_tm_10 + (i * 3 + k) * in_c;
                    const __fp16 g0 = G[j][k];
                    vfloat16m1_t _g0 = vfmv_v_f_f16m1(g0, vl_16);
                    int c = 0;
                    for (; c + vl_16 - 1 < in_c; c += vl_16) {
                        const __fp16 *tmp0 = tmp_ptr + c;
                        __fp16 *res0 = res_ptr + c;
                        vfloat16m1_t _res = vle16_v_f16m1(res0, vl_16);
                        vfloat16m1_t _kernel = vle16_v_f16m1(tmp0, vl_16);
                        _res = vfmacc_vv_f16m1(_res, _g0, _kernel, vl_16);
                        vse16_v_f16m1(res0, _res, vl_16);
                    }
                    while (c < in_c) {
                        const __fp16 *tmp0 = tmp_ptr + c;
                        __fp16 *res0 = res_ptr + c;
                        vl = vsetvl_e16m1(in_c - c);
                        vfloat16m1_t _res = vle16_v_f16m1(res0, vl);
                        vfloat16m1_t _kernel = vle16_v_f16m1(tmp0, vl);
                        _res = vfmacc_vf_f16m1(_res, g0, _kernel, vl);
                        vse16_v_f16m1(res0, _res, vl);
                        c += vl;
                    }
                }
            }
        }
    }
    shl_mem_free(kernel_tm_1);

    // optimized layout for winograd64
    // [O, 6, 6, I] --> [36, O, I]
    __fp16 *kernel_tm_3 = (__fp16 *)shl_mem_alloc(36 * out_c * in_c * sizeof(__fp16));
    for (int a = 0; a < 36; a++) {
        for (int oc = 0; oc < out_c; oc++) {
            const __fp16 *src_ptr = kernel_tm_2 + (oc * 36 + a) * in_c;
            __fp16 *dst_ptr = kernel_tm_3 + (a * out_c + oc) * in_c;
            int c = 0;
            while (c < in_c) {
                const __fp16 *src0 = src_ptr + c;
                __fp16 *dst0 = dst_ptr + c;
                vl = vsetvl_e16m1(in_c - c);
                vfloat16m1_t _src = vle16_v_f16m1(src0, vl);
                vse16_v_f16m1(dst0, _src, vl);
                c += vl;
            }
        }
    }
    shl_mem_free(kernel_tm_2);

    // reorder & align kernel
    int k_align = ((in_c - 1) & -(csrr_xmlenb() / 2)) + (csrr_xmlenb() / 2);
    int n_align = ((out_c - 1) & -(csrr_xmlenb() / 2)) + (csrr_xmlenb() / 2);
    __fp16 *kernel_reorder = (__fp16 *)shl_mem_alloc(36 * n_align * k_align * sizeof(__fp16));
    const int mcols = csrr_xmlenb() / 2;
    for (int a = 0; a < 36; a++) {
        const __fp16 *src = kernel_tm_3 + a * out_c * in_c;
        __fp16 *dst = kernel_reorder + a * n_align * k_align;
        wg_bxf3s1_reorder_kernel_nhwc_fp16(dst, src, out_c, in_c, mcols);
    }

    dst_kernel->data = kernel_reorder;
}

/******************************************************************************************
 * kernel layout before:  [O, 3, 3, I]
 * kernel layout after :  [O, 8, 8, I] --> [64, O, I]
 ******************************************************************************************/
void shl_rvm_wg_b6f3s1_trans_kernel_nhwc_fp16(struct csinn_tensor *src_kernel,
                                              struct csinn_tensor *dst_kernel)
{
    int32_t out_c = src_kernel->dim[0];
    int32_t in_c = src_kernel->dim[3];
    __fp16 *kernel_data = (__fp16 *)src_kernel->data;
    /* kernel transform matrix: G
    G =
        ⎡ 1       0      0  ⎤
        ⎢-2/9   -2/9    -2/9⎥
        ⎢-2/9    2/9    -2/9⎥
        ⎢1/90   1/45    2/45⎥
        ⎢1/90   -1/45   2/45⎥
        ⎢─32/45  16/45  8/45⎥
        ⎢32/45  -16/45  8/45⎥
        ⎣ 0       0      1  ⎦
    */
    const __fp16 G[8][3] = {{1.0f, 0.0f, 0.0f},
                            {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                            {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                            {1.0f / 90, 1.0f / 45, 2.0f / 45},
                            {1.0f / 90, -1.0f / 45, 2.0f / 45},
                            {1.0f / 45, 1.0f / 90, 1.0f / 180},
                            {1.0f / 45, -1.0f / 90, 1.0f / 180},
                            {0.0f, 0.0f, 1.0f}};

    csinn_tensor_copy(dst_kernel, src_kernel);

    // for kernel transform buf, 3x3 --> 8x8
    __fp16 *kernel_tm_1 = (__fp16 *)shl_mem_alloc(out_c * 8 * 3 * in_c * sizeof(__fp16));
    __fp16 *kernel_tm_2 = (__fp16 *)shl_mem_alloc(out_c * 8 * 8 * in_c * sizeof(__fp16));

    const int vl_16 = vsetvl_e16m1(in_c);
    int vl = 0;
    for (int oc = 0; oc < out_c; oc++) {
        // G * g
        __fp16 *kernel_tm_10 = kernel_tm_1 + oc * 8 * 3 * in_c;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 3; j++) {
                __fp16 *res_ptr = kernel_tm_10 + (i * 3 + j) * in_c;
                for (int k = 0; k < 3; k++) {
                    const __fp16 *k_ptr = kernel_data + ((oc * 3 + k) * 3 + j) * in_c;
                    const __fp16 g0 = G[i][k];
                    vfloat16m1_t _g0 = vfmv_v_f_f16m1(g0, vl_16);
                    int c = 0;
                    for (; c + vl_16 - 1 < in_c; c += vl_16) {
                        const __fp16 *k0 = k_ptr + c;
                        __fp16 *res0 = res_ptr + c;
                        vfloat16m1_t _res = vle16_v_f16m1(res0, vl_16);
                        vfloat16m1_t _kernel = vle16_v_f16m1(k0, vl_16);
                        _res = vfmacc_vv_f16m1(_res, _g0, _kernel, vl_16);
                        vse16_v_f16m1(res0, _res, vl_16);
                    }
                    while (c < in_c) {
                        const __fp16 *k0 = k_ptr + c;
                        __fp16 *res0 = res_ptr + c;
                        vl = vsetvl_e16m1(in_c - c);
                        vfloat16m1_t _res = vle16_v_f16m1(res0, vl);
                        vfloat16m1_t _kernel = vle16_v_f16m1(k0, vl);
                        _res = vfmacc_vf_f16m1(_res, g0, _kernel, vl);
                        vse16_v_f16m1(res0, _res, vl);
                        c += vl;
                    }
                }
            }
        }

        // * GT
        __fp16 *kernel_tm_20 = kernel_tm_2 + oc * 64 * in_c;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                __fp16 *res_ptr = kernel_tm_20 + (i * 8 + j) * in_c;
                for (int k = 0; k < 3; k++) {
                    const __fp16 *tmp_ptr = kernel_tm_10 + (i * 3 + k) * in_c;
                    const __fp16 g0 = G[j][k];
                    vfloat16m1_t _g0 = vfmv_v_f_f16m1(g0, vl_16);
                    int c = 0;
                    for (; c + vl_16 - 1 < in_c; c += vl_16) {
                        const __fp16 *tmp0 = tmp_ptr + c;
                        __fp16 *res0 = res_ptr + c;
                        vfloat16m1_t _res = vle16_v_f16m1(res0, vl_16);
                        vfloat16m1_t _kernel = vle16_v_f16m1(tmp0, vl_16);
                        _res = vfmacc_vv_f16m1(_res, _g0, _kernel, vl_16);
                        vse16_v_f16m1(res0, _res, vl_16);
                    }
                    while (c < in_c) {
                        const __fp16 *tmp0 = tmp_ptr + c;
                        __fp16 *res0 = res_ptr + c;
                        vl = vsetvl_e16m1(in_c - c);
                        vfloat16m1_t _res = vle16_v_f16m1(res0, vl);
                        vfloat16m1_t _kernel = vle16_v_f16m1(tmp0, vl);
                        _res = vfmacc_vf_f16m1(_res, g0, _kernel, vl);
                        vse16_v_f16m1(res0, _res, vl);
                        c += vl;
                    }
                }
            }
        }
    }
    shl_mem_free(kernel_tm_1);

    // optimized layout for winograd64
    // [O, 8, 8, I] --> [64, O, I]
    __fp16 *kernel_tm_3 = (__fp16 *)shl_mem_alloc(64 * out_c * in_c * sizeof(__fp16));
    for (int a = 0; a < 64; a++) {
        for (int oc = 0; oc < out_c; oc++) {
            const __fp16 *src_ptr = kernel_tm_2 + (oc * 64 + a) * in_c;
            __fp16 *dst_ptr = kernel_tm_3 + (a * out_c + oc) * in_c;
            int c = 0;
            while (c < in_c) {
                const __fp16 *src0 = src_ptr + c;
                __fp16 *dst0 = dst_ptr + c;
                vl = vsetvl_e16m1(in_c - c);
                vfloat16m1_t _src = vle16_v_f16m1(src0, vl);
                vse16_v_f16m1(dst0, _src, vl);
                c += vl;
            }
        }
    }
    shl_mem_free(kernel_tm_2);

    // reorder & align kernel
    int k_align = ((in_c - 1) & -(csrr_xmlenb() / 2)) + (csrr_xmlenb() / 2);
    int n_align = ((out_c - 1) & -(csrr_xmlenb() / 2)) + (csrr_xmlenb() / 2);
    __fp16 *kernel_reorder = (__fp16 *)shl_mem_alloc(64 * n_align * k_align * sizeof(__fp16));
    const int mcols = csrr_xmlenb() / 2;
    for (int a = 0; a < 64; a++) {
        const __fp16 *src = kernel_tm_3 + a * out_c * in_c;
        __fp16 *dst = kernel_reorder + a * n_align * k_align;
        wg_bxf3s1_reorder_kernel_nhwc_fp16(dst, src, out_c, in_c, mcols);
    }

    dst_kernel->data = kernel_reorder;
}

static void winograd_pad_input_nhwc_fp16(const __fp16 *input, __fp16 *input_padded, int inh,
                                         int inw, int inc, int padded_h, int padded_w, int pad_top,
                                         int pad_left)
{
    shl_rvv_pad_input_nhwc_fp16(input, input_padded, inh, inw, inc, padded_h, padded_w, pad_top,
                                pad_left);
}

static inline void wg_bxf3s1_batch_gemm_matrix_fp16(const __fp16 *input, const __fp16 *kernel,
                                                    __fp16 *output, __fp16 *fake_bias, int k_align,
                                                    int out_c, int tiles, int area)
{
    int n_align = ((out_c - 1) & -(csrr_xmlenb() / 2)) + (csrr_xmlenb() / 2);
    for (int r = 0; r < area; r++) {
        const __fp16 *input_ptr = input + r * tiles * k_align;
        const __fp16 *kernel_ptr = kernel + r * n_align * k_align;
        __fp16 *output_ptr = output + r * tiles * out_c;

        shl_rvm_nhwc_gemm_fp16(output_ptr, kernel_ptr, input_ptr, fake_bias, tiles, k_align, out_c);
    }
}

static void winograd_crop_output_nhwc_fp16(const __fp16 *output_trans, __fp16 *output, int out_h,
                                           int out_w, int out_c, int wino_h, int wino_w)
{
    int vl = 0;
    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            const __fp16 *src_ptr = output_trans + (h * wino_w + w) * out_c;
            __fp16 *dst_ptr = output + (h * out_w + w) * out_c;
            int c = 0;
            while (c < out_c) {
                const __fp16 *src0 = src_ptr + c;
                __fp16 *dst0 = dst_ptr + c;
                vl = vsetvl_e16m1(out_c - c);
                vfloat16m1_t _src = vle16_v_f16m1(src0, vl);
                vse16_v_f16m1(dst0, _src, vl);
                c += vl;
            }
        }
    }
}

/******************************************************************************************
 * input layout before:  [blk_h*4+2, blk_w*4+2, in_c]
 * input layout after :  [tiles, 36, in_c] --> [36, tiles, k_align]
 ******************************************************************************************/
static inline void wg_b4f3s1_trans_input_nhwc_fp16(const __fp16 *src, __fp16 *dst, int in_h,
                                                   int in_w, int in_c, int blk_h, int blk_w)
{
    /* input transform matrix
    BT = {
        { 4   0   -5   0   1  0 };
        { 0  -4   -4   1   1  0 };
        { 0   4   -4  -1   1  0 };
        { 0  -2   -1   2   1  0 };
        { 0   2   -1  -2   1  0 };
        { 0   4    0  -5   0  1 }
    };
    */
    int tiles = blk_h * blk_w;
    int vl = vsetvl_e16m1(in_c);
    __fp16 tmp[6][6][vl];
    int k_align = ((in_c - 1) & -(csrr_xmlenb() / 2)) + (csrr_xmlenb() / 2);

    for (int i = 0; i < blk_h; i++) {
        for (int j = 0; j < blk_w; j++) {
            const __fp16 *img = src + (i * 4 * in_w + j * 4) * in_c;
            __fp16 *img_tm = dst + (i * blk_w + j) * k_align;
            int c = 0;
            while (c < in_c) {
                const __fp16 *r0 = img + c;
                __fp16 *r0_tm = img_tm + c;
                vl = vsetvl_e16m1(in_c - c);
                for (int m = 0; m < 6; m++) {
                    vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                    vfloat16m1_t _r01 = vle16_v_f16m1(r0 + in_c * 1, vl);
                    vfloat16m1_t _r02 = vle16_v_f16m1(r0 + in_c * 2, vl);
                    vfloat16m1_t _r03 = vle16_v_f16m1(r0 + in_c * 3, vl);
                    vfloat16m1_t _r04 = vle16_v_f16m1(r0 + in_c * 4, vl);
                    vfloat16m1_t _r05 = vle16_v_f16m1(r0 + in_c * 5, vl);

                    vfloat16m1_t _tmp0m =
                        vfmacc_vf_f16m1(vfmacc_vf_f16m1(_r04, 4.f, _r00, vl), -5.f, _r02, vl);
                    vfloat16m1_t _tmp1m = vfmacc_vf_f16m1(vfadd_vv_f16m1(_r04, _r03, vl), -4.f,
                                                          vfadd_vv_f16m1(_r01, _r02, vl), vl);
                    vfloat16m1_t _tmp2m = vfmacc_vf_f16m1(vfsub_vv_f16m1(_r04, _r03, vl), 4.f,
                                                          vfsub_vv_f16m1(_r01, _r02, vl), vl);
                    vfloat16m1_t _tmp3m = vfmacc_vf_f16m1(vfsub_vv_f16m1(_r04, _r02, vl), -2.f,
                                                          vfsub_vv_f16m1(_r01, _r03, vl), vl);
                    vfloat16m1_t _tmp4m = vfmacc_vf_f16m1(vfsub_vv_f16m1(_r04, _r02, vl), 2.f,
                                                          vfsub_vv_f16m1(_r01, _r03, vl), vl);
                    vfloat16m1_t _tmp5m =
                        vfmacc_vf_f16m1(vfmacc_vf_f16m1(_r05, 4.f, _r01, vl), -5.f, _r03, vl);

                    vse16_v_f16m1(tmp[0][m], _tmp0m, vl);
                    vse16_v_f16m1(tmp[1][m], _tmp1m, vl);
                    vse16_v_f16m1(tmp[2][m], _tmp2m, vl);
                    vse16_v_f16m1(tmp[3][m], _tmp3m, vl);
                    vse16_v_f16m1(tmp[4][m], _tmp4m, vl);
                    vse16_v_f16m1(tmp[5][m], _tmp5m, vl);

                    r0 += in_w * in_c;
                }
                for (int m = 0; m < 6; m++) {
                    __fp16 *r0_tm0 = r0_tm;
                    __fp16 *r0_tm1 = r0_tm0 + 6 * tiles * k_align;
                    __fp16 *r0_tm2 = r0_tm1 + 6 * tiles * k_align;
                    __fp16 *r0_tm3 = r0_tm2 + 6 * tiles * k_align;
                    __fp16 *r0_tm4 = r0_tm3 + 6 * tiles * k_align;
                    __fp16 *r0_tm5 = r0_tm4 + 6 * tiles * k_align;

                    vfloat16m1_t _tmp00 = vle16_v_f16m1(tmp[m][0], vl);
                    vfloat16m1_t _tmp01 = vle16_v_f16m1(tmp[m][1], vl);
                    vfloat16m1_t _tmp02 = vle16_v_f16m1(tmp[m][2], vl);
                    vfloat16m1_t _tmp03 = vle16_v_f16m1(tmp[m][3], vl);
                    vfloat16m1_t _tmp04 = vle16_v_f16m1(tmp[m][4], vl);
                    vfloat16m1_t _tmp05 = vle16_v_f16m1(tmp[m][5], vl);

                    vfloat16m1_t _r0tm0 =
                        vfmacc_vf_f16m1(vfmacc_vf_f16m1(_tmp04, 4.f, _tmp00, vl), -5.f, _tmp02, vl);
                    vfloat16m1_t _r0tm1 = vfmacc_vf_f16m1(vfadd_vv_f16m1(_tmp04, _tmp03, vl), -4.f,
                                                          vfadd_vv_f16m1(_tmp01, _tmp02, vl), vl);
                    vfloat16m1_t _r0tm2 = vfmacc_vf_f16m1(vfsub_vv_f16m1(_tmp04, _tmp03, vl), 4.f,
                                                          vfsub_vv_f16m1(_tmp01, _tmp02, vl), vl);
                    vfloat16m1_t _r0tm3 = vfmacc_vf_f16m1(vfsub_vv_f16m1(_tmp04, _tmp02, vl), -2.f,
                                                          vfsub_vv_f16m1(_tmp01, _tmp03, vl), vl);
                    vfloat16m1_t _r0tm4 = vfmacc_vf_f16m1(vfsub_vv_f16m1(_tmp04, _tmp02, vl), 2.f,
                                                          vfsub_vv_f16m1(_tmp01, _tmp03, vl), vl);
                    vfloat16m1_t _r0tm5 =
                        vfmacc_vf_f16m1(vfmacc_vf_f16m1(_tmp05, 4.f, _tmp01, vl), -5.f, _tmp03, vl);

                    vse16_v_f16m1(r0_tm0, _r0tm0, vl);
                    vse16_v_f16m1(r0_tm1, _r0tm1, vl);
                    vse16_v_f16m1(r0_tm2, _r0tm2, vl);
                    vse16_v_f16m1(r0_tm3, _r0tm3, vl);
                    vse16_v_f16m1(r0_tm4, _r0tm4, vl);
                    vse16_v_f16m1(r0_tm5, _r0tm5, vl);

                    r0_tm += tiles * k_align;
                }
                c += vl;
            }
        }
    }
}

/******************************************************************************************
 * output layout before:  [36, tiles, out_c]
 * output layout after :  [16, tiles, out_c] --> [blk_h*4, blk_w*4, out_c]
 ******************************************************************************************/
static inline void wg_b4f3s1_trans_output_nhwc_fp16(const __fp16 *src, const __fp16 *bias,
                                                    __fp16 *dst, int blk_h, int blk_w, int out_c)
{
    /* output transform matrix
    AT = {
        { 1  1  1   1  1   0 },
        { 0  1  -1  2  -2  0 },
        { 0  1  1   4  4   0 },
        { 0  1  -1  8  -8  1 }
    };
    */
    int tiles = blk_h * blk_w;
    int vl = vsetvl_e16m1(out_c);
    __fp16 tmp[4][6][vl];

    for (int i = 0; i < blk_h; i++) {
        for (int j = 0; j < blk_w; j++) {
            int c = 0;
            while (c < out_c) {
                vl = vsetvl_e16m1(out_c - c);

                const __fp16 *output0_tm_0 = src + (i * blk_w + j) * out_c + c;  // 6*6 起始地址
                const __fp16 *output0_tm_1 = output0_tm_0 + tiles * out_c * 6;
                const __fp16 *output0_tm_2 = output0_tm_1 + tiles * out_c * 6;
                const __fp16 *output0_tm_3 = output0_tm_2 + tiles * out_c * 6;
                const __fp16 *output0_tm_4 = output0_tm_3 + tiles * out_c * 6;
                const __fp16 *output0_tm_5 = output0_tm_4 + tiles * out_c * 6;
                __fp16 *output0 = dst + (i * 4 * blk_w * 4 + j * 4) * out_c + c;  // out 4*4 addr

                vfloat16m1_t _bias = bias ? vle16_v_f16m1(bias + c, vl) : vfmv_v_f_f16m1(0.0f, vl);

                for (int m = 0; m < 6; m++) {
                    vfloat16m1_t _r00 = vle16_v_f16m1(output0_tm_0, vl);
                    vfloat16m1_t _r01 = vle16_v_f16m1(output0_tm_1, vl);
                    vfloat16m1_t _r02 = vle16_v_f16m1(output0_tm_2, vl);
                    vfloat16m1_t _r03 = vle16_v_f16m1(output0_tm_3, vl);
                    vfloat16m1_t _r04 = vle16_v_f16m1(output0_tm_4, vl);
                    vfloat16m1_t _r05 = vle16_v_f16m1(output0_tm_5, vl);

                    vfloat16m1_t _tmp02a = vfadd_vv_f16m1(_r01, _r02, vl);
                    vfloat16m1_t _tmp13a = vfsub_vv_f16m1(_r01, _r02, vl);

                    vfloat16m1_t _tmp02b = vfadd_vv_f16m1(_r03, _r04, vl);
                    vfloat16m1_t _tmp13b = vfsub_vv_f16m1(_r03, _r04, vl);

                    vfloat16m1_t _tmp0m =
                        vfadd_vv_f16m1(vfadd_vv_f16m1(_r00, _tmp02a, vl), _tmp02b, vl);
                    vfloat16m1_t _tmp1m = vfmacc_vf_f16m1(_tmp13a, 2.f, _tmp13b, vl);
                    vfloat16m1_t _tmp2m = vfmacc_vf_f16m1(_tmp02a, 4.f, _tmp02b, vl);
                    vfloat16m1_t _tmp3m =
                        vfmacc_vf_f16m1(vfadd_vv_f16m1(_r05, _tmp13a, vl), 8.f, _tmp13b, vl);

                    vse16_v_f16m1(tmp[0][m], _tmp0m, vl);
                    vse16_v_f16m1(tmp[1][m], _tmp1m, vl);
                    vse16_v_f16m1(tmp[2][m], _tmp2m, vl);
                    vse16_v_f16m1(tmp[3][m], _tmp3m, vl);

                    output0_tm_0 += tiles * out_c;
                    output0_tm_1 += tiles * out_c;
                    output0_tm_2 += tiles * out_c;
                    output0_tm_3 += tiles * out_c;
                    output0_tm_4 += tiles * out_c;
                    output0_tm_5 += tiles * out_c;
                }
                for (int m = 0; m < 4; m++) {
                    vfloat16m1_t _tmp00 = vle16_v_f16m1(tmp[m][0], vl);
                    vfloat16m1_t _tmp01 = vle16_v_f16m1(tmp[m][1], vl);
                    vfloat16m1_t _tmp02 = vle16_v_f16m1(tmp[m][2], vl);
                    vfloat16m1_t _tmp03 = vle16_v_f16m1(tmp[m][3], vl);
                    vfloat16m1_t _tmp04 = vle16_v_f16m1(tmp[m][4], vl);
                    vfloat16m1_t _tmp05 = vle16_v_f16m1(tmp[m][5], vl);

                    vfloat16m1_t _tmp02a = vfadd_vv_f16m1(_tmp01, _tmp02, vl);
                    vfloat16m1_t _tmp13a = vfsub_vv_f16m1(_tmp01, _tmp02, vl);

                    vfloat16m1_t _tmp02b = vfadd_vv_f16m1(_tmp03, _tmp04, vl);
                    vfloat16m1_t _tmp13b = vfsub_vv_f16m1(_tmp03, _tmp04, vl);

                    vfloat16m1_t _out00 =
                        vfadd_vv_f16m1(vfadd_vv_f16m1(_tmp00, _tmp02a, vl), _tmp02b, vl);
                    vfloat16m1_t _out01 = vfmacc_vf_f16m1(_tmp13a, 2.f, _tmp13b, vl);
                    vfloat16m1_t _out02 = vfmacc_vf_f16m1(_tmp02a, 4.f, _tmp02b, vl);
                    vfloat16m1_t _out03 =
                        vfmacc_vf_f16m1(vfadd_vv_f16m1(_tmp05, _tmp13a, vl), 8.f, _tmp13b, vl);

                    _out00 = vfadd_vv_f16m1(_bias, _out00, vl);
                    _out01 = vfadd_vv_f16m1(_bias, _out01, vl);
                    _out02 = vfadd_vv_f16m1(_bias, _out02, vl);
                    _out03 = vfadd_vv_f16m1(_bias, _out03, vl);

                    vse16_v_f16m1(output0, _out00, vl);
                    vse16_v_f16m1(output0 + out_c * 1, _out01, vl);
                    vse16_v_f16m1(output0 + out_c * 2, _out02, vl);
                    vse16_v_f16m1(output0 + out_c * 3, _out03, vl);

                    output0 += blk_w * 4 * out_c;
                }
                c += vl;
            }
        }
    }
}

int shl_rvm_wg_b4f3s1_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)params->conv_extra.kernel_tm->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int pad_left = params->pad_left;
    int pad_top = params->pad_top;

    int batch = input->dim[0];
    int in_h = input->dim[1];
    int in_w = input->dim[2];
    int in_c = input->dim[3];
    int input_size = in_c * in_h * in_w;

    int out_c = kernel->dim[0];
    int out_h = output->dim[1];
    int out_w = output->dim[2];
    int output_size = out_c * out_h * out_w;

    int block_h = (out_h + 3) / 4;
    int block_w = (out_w + 3) / 4;

    // block * 4 for alignment with 4, kernel = 3 * 3, stride = 1, thus input_size + 2
    int padded_in_h = block_h * 4 + 2;
    int padded_in_w = block_w * 4 + 2;
    int padded_in_hw = padded_in_h * padded_in_w;  // element size after padding per channel

    int tiles = block_h * block_w;
    int k_align = ((in_c - 1) & -(csrr_xmlenb() / 2)) + (csrr_xmlenb() / 2);

    for (int b = 0; b < batch; b++) {
        /****************************** pad input *****************************/
        // pad buffer: [blk_h*4+2, blk_w*4+2, in_c]
        __fp16 *input_padd_buf = (__fp16 *)shl_mem_alloc(padded_in_hw * in_c * sizeof(__fp16));
        winograd_pad_input_nhwc_fp16(input_data, input_padd_buf, in_h, in_w, in_c, padded_in_h,
                                     padded_in_w, pad_top, pad_left);
        input_data += input_size;

        /****************************** transform input *****************************/
        // input_tm_buf: [36, tiles, in_c]
        __fp16 *input_tm_buf = (__fp16 *)shl_mem_alloc(36 * tiles * k_align * sizeof(__fp16));
        wg_b4f3s1_trans_input_nhwc_fp16(input_padd_buf, input_tm_buf, padded_in_h, padded_in_w,
                                        in_c, block_h, block_w);
        shl_mem_free(input_padd_buf);

        /****************************** batch gemm *****************************/
        // output_dot_buf: [36, tiles, out_c]
        __fp16 *output_dot_buf = (__fp16 *)shl_mem_alloc(36 * tiles * out_c * sizeof(__fp16));
        wg_bxf3s1_batch_gemm_matrix_fp16(input_tm_buf, kernel_data, output_dot_buf, NULL, k_align,
                                         out_c, tiles, 36);
        shl_mem_free(input_tm_buf);

        /****************************** transform output *****************************/
        // output_tm_buf: [blk_h*4, blk_w*4, out_c]
        __fp16 *output_tm_buf =
            (__fp16 *)shl_mem_alloc(block_h * 4 * block_w * 4 * out_c * sizeof(__fp16));
        wg_b4f3s1_trans_output_nhwc_fp16(output_dot_buf, bias_data, output_tm_buf, block_h, block_w,
                                         out_c);
        shl_mem_free(output_dot_buf);

        /****************************** crop output *****************************/
        // crop the output after transform: cut extra part (right , bottom)
        winograd_crop_output_nhwc_fp16(output_tm_buf, output_data, out_h, out_w, out_c, block_h * 4,
                                       block_w * 4);

        output_data += output_size;
        shl_mem_free(output_tm_buf);
    }
    return CSINN_TRUE;
}

/******************************************************************************************
 * input layout before:  [blk_h*6+2, blk_w*6+2, in_c]
 * input layout after :  [tiles, 64, in_c] --> [64, tiles, k_align]
 ******************************************************************************************/
static inline void wg_b6f3s1_trans_input_nhwc_fp16(const __fp16 *src, __fp16 *dst, int in_h,
                                                   int in_w, int in_c, int blk_h, int blk_w)
{
    /* input transform matrix
    BT = {
        { 1   0    -5.25    0    5.25     0    -1  0 };
        { 0   1      1    -4.25  -4.25    1    1   0 };
        { 0   -1     1    4.25   -4.25   -1    1   0 };
        { 0  0.5    0.25   -2.5   -1.25     2    1   0 };
        { 0  -0.5   0.25    2.5   -1.25    -2    1   0 };
        { 0   2      4    -2.5    -5     0.5   1   0 };
        { 0   -2     4     2.5    -5    -0.5   1   0 };
        { 0   -1     0    5.25     0    -5.25  0   1 }
    };
    */
    int tiles = blk_h * blk_w;
    int vl = vsetvl_e16m1(in_c);
    __fp16 tmp[8][8][vl];
    int k_align = ((in_c - 1) & -(csrr_xmlenb() / 2)) + (csrr_xmlenb() / 2);

    for (int i = 0; i < blk_h; i++) {
        for (int j = 0; j < blk_w; j++) {
            const __fp16 *img = src + (i * 6 * in_w + j * 6) * in_c;
            __fp16 *img_tm = dst + (i * blk_w + j) * k_align;
            int c = 0;
            while (c < in_c) {
                vl = vsetvl_e16m1(in_c - c);
                const __fp16 *r0 = img + c;
                __fp16 *r0_tm = img_tm + c;
                for (int m = 0; m < 8; m++) {
                    vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                    vfloat16m1_t _r01 = vle16_v_f16m1(r0 + in_c * 1, vl);
                    vfloat16m1_t _r02 = vle16_v_f16m1(r0 + in_c * 2, vl);
                    vfloat16m1_t _r03 = vle16_v_f16m1(r0 + in_c * 3, vl);
                    vfloat16m1_t _r04 = vle16_v_f16m1(r0 + in_c * 4, vl);
                    vfloat16m1_t _r05 = vle16_v_f16m1(r0 + in_c * 5, vl);
                    vfloat16m1_t _r06 = vle16_v_f16m1(r0 + in_c * 6, vl);
                    vfloat16m1_t _r07 = vle16_v_f16m1(r0 + in_c * 7, vl);

                    vfloat16m1_t _tmp0m = vfmacc_vf_f16m1(vfsub_vv_f16m1(_r00, _r06, vl), 5.25f,
                                                          vfsub_vv_f16m1(_r04, _r02, vl), vl);
                    vfloat16m1_t _tmp7m = vfmacc_vf_f16m1(vfsub_vv_f16m1(_r07, _r01, vl), 5.25f,
                                                          vfsub_vv_f16m1(_r03, _r05, vl), vl);

                    vfloat16m1_t _tmp12a =
                        vfmacc_vf_f16m1(vfadd_vv_f16m1(_r02, _r06, vl), -4.25f, _r04, vl);
                    vfloat16m1_t _tmp12b =
                        vfmacc_vf_f16m1(vfadd_vv_f16m1(_r01, _r05, vl), -4.25f, _r03, vl);
                    vfloat16m1_t _tmp1m = vfadd_vv_f16m1(_tmp12a, _tmp12b, vl);
                    vfloat16m1_t _tmp2m = vfsub_vv_f16m1(_tmp12a, _tmp12b, vl);

                    vfloat16m1_t _tmp34a =
                        vfmacc_vf_f16m1(vfmacc_vf_f16m1(_r06, 0.25f, _r02, vl), -1.25f, _r04, vl);
                    vfloat16m1_t _tmp34b = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(vfmul_vf_f16m1(_r01, 0.5f, vl), -2.5f, _r03, vl), 2.f, _r05,
                        vl);
                    vfloat16m1_t _tmp3m = vfadd_vv_f16m1(_tmp34a, _tmp34b, vl);
                    vfloat16m1_t _tmp4m = vfsub_vv_f16m1(_tmp34a, _tmp34b, vl);

                    vfloat16m1_t _tmp56a =
                        vfmacc_vf_f16m1(_r06, 4.f, vfmacc_vf_f16m1(_r02, -1.25f, _r04, vl), vl);
                    vfloat16m1_t _tmp56b = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(vfmul_vf_f16m1(_r01, 2.f, vl), -2.5f, _r03, vl), 0.5f, _r05,
                        vl);
                    vfloat16m1_t _tmp5m = vfadd_vv_f16m1(_tmp56a, _tmp56b, vl);
                    vfloat16m1_t _tmp6m = vfsub_vv_f16m1(_tmp56a, _tmp56b, vl);

                    vse16_v_f16m1(tmp[0][m], _tmp0m, vl);
                    vse16_v_f16m1(tmp[7][m], _tmp7m, vl);
                    vse16_v_f16m1(tmp[1][m], _tmp1m, vl);
                    vse16_v_f16m1(tmp[2][m], _tmp2m, vl);
                    vse16_v_f16m1(tmp[3][m], _tmp3m, vl);
                    vse16_v_f16m1(tmp[4][m], _tmp4m, vl);
                    vse16_v_f16m1(tmp[5][m], _tmp5m, vl);
                    vse16_v_f16m1(tmp[6][m], _tmp6m, vl);

                    r0 += in_w * in_c;
                }
                for (int m = 0; m < 8; m++) {
                    __fp16 *r0_tm0 = r0_tm;
                    __fp16 *r0_tm1 = r0_tm0 + 8 * tiles * k_align;
                    __fp16 *r0_tm2 = r0_tm1 + 8 * tiles * k_align;
                    __fp16 *r0_tm3 = r0_tm2 + 8 * tiles * k_align;
                    __fp16 *r0_tm4 = r0_tm3 + 8 * tiles * k_align;
                    __fp16 *r0_tm5 = r0_tm4 + 8 * tiles * k_align;
                    __fp16 *r0_tm6 = r0_tm5 + 8 * tiles * k_align;
                    __fp16 *r0_tm7 = r0_tm6 + 8 * tiles * k_align;

                    vfloat16m1_t _tmp00 = vle16_v_f16m1(tmp[m][0], vl);
                    vfloat16m1_t _tmp01 = vle16_v_f16m1(tmp[m][1], vl);
                    vfloat16m1_t _tmp02 = vle16_v_f16m1(tmp[m][2], vl);
                    vfloat16m1_t _tmp03 = vle16_v_f16m1(tmp[m][3], vl);
                    vfloat16m1_t _tmp04 = vle16_v_f16m1(tmp[m][4], vl);
                    vfloat16m1_t _tmp05 = vle16_v_f16m1(tmp[m][5], vl);
                    vfloat16m1_t _tmp06 = vle16_v_f16m1(tmp[m][6], vl);
                    vfloat16m1_t _tmp07 = vle16_v_f16m1(tmp[m][7], vl);

                    vfloat16m1_t _r0tm0 = vfmacc_vf_f16m1(vfsub_vv_f16m1(_tmp00, _tmp06, vl), 5.25f,
                                                          vfsub_vv_f16m1(_tmp04, _tmp02, vl), vl);
                    vfloat16m1_t _r0tm7 = vfmacc_vf_f16m1(vfsub_vv_f16m1(_tmp07, _tmp01, vl), 5.25f,
                                                          vfsub_vv_f16m1(_tmp03, _tmp05, vl), vl);

                    vfloat16m1_t _tmp12a =
                        vfmacc_vf_f16m1(vfadd_vv_f16m1(_tmp02, _tmp06, vl), -4.25f, _tmp04, vl);
                    vfloat16m1_t _tmp12b =
                        vfmacc_vf_f16m1(vfadd_vv_f16m1(_tmp01, _tmp05, vl), -4.25f, _tmp03, vl);
                    vfloat16m1_t _r0tm1 = vfadd_vv_f16m1(_tmp12a, _tmp12b, vl);
                    vfloat16m1_t _r0tm2 = vfsub_vv_f16m1(_tmp12a, _tmp12b, vl);

                    vfloat16m1_t _tmp34a = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(_tmp06, 0.25f, _tmp02, vl), -1.25f, _tmp04, vl);
                    vfloat16m1_t _tmp34b = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(vfmul_vf_f16m1(_tmp01, 0.5f, vl), -2.5f, _tmp03, vl), 2.f,
                        _tmp05, vl);
                    vfloat16m1_t _r0tm3 = vfadd_vv_f16m1(_tmp34a, _tmp34b, vl);
                    vfloat16m1_t _r0tm4 = vfsub_vv_f16m1(_tmp34a, _tmp34b, vl);

                    vfloat16m1_t _tmp56a = vfmacc_vf_f16m1(
                        _tmp06, 4.f, vfmacc_vf_f16m1(_tmp02, -1.25f, _tmp04, vl), vl);
                    vfloat16m1_t _tmp56b = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(vfmul_vf_f16m1(_tmp01, 2.f, vl), -2.5f, _tmp03, vl), 0.5f,
                        _tmp05, vl);
                    vfloat16m1_t _r0tm5 = vfadd_vv_f16m1(_tmp56a, _tmp56b, vl);
                    vfloat16m1_t _r0tm6 = vfsub_vv_f16m1(_tmp56a, _tmp56b, vl);

                    vse16_v_f16m1(r0_tm0, _r0tm0, vl);
                    vse16_v_f16m1(r0_tm7, _r0tm7, vl);
                    vse16_v_f16m1(r0_tm1, _r0tm1, vl);
                    vse16_v_f16m1(r0_tm2, _r0tm2, vl);
                    vse16_v_f16m1(r0_tm3, _r0tm3, vl);
                    vse16_v_f16m1(r0_tm4, _r0tm4, vl);
                    vse16_v_f16m1(r0_tm5, _r0tm5, vl);
                    vse16_v_f16m1(r0_tm6, _r0tm6, vl);

                    r0_tm += tiles * k_align;
                }
                c += vl;
            }
        }
    }
}

/******************************************************************************************
 * output layout before:  [64, tiles, out_c]
 * output layout after :  [36, tiles, out_c] --> [blk_h*6, blk_w*6, out_c]
 ******************************************************************************************/
static inline void wg_b6f3s1_trans_output_nhwc_fp16(const __fp16 *src, const __fp16 *bias,
                                                    __fp16 *dst, int blk_h, int blk_w, int out_c)
{
    /* output transform matrix
    AT = {
        { 1  1  1   1    1    1      1    0 };
        { 0  1  -1  2   -2   1/2   -1/2   0 };
        { 0  1  1   4    4   1/4    1/4   0 };
        { 0  1  -1  8   -8   1/8   -1/8   0 };
        { 0  1  1   16  16   1/16  1/16   0 };
        { 0  1  -1  32  -32  1/32  -1/32  1 }
    };
    */
    int tiles = blk_h * blk_w;
    int vl = vsetvl_e16m1(out_c);
    __fp16 tmp[6][8][vl];

    for (int i = 0; i < blk_h; i++) {
        for (int j = 0; j < blk_w; j++) {
            int c = 0;
            while (c < out_c) {
                vl = vsetvl_e16m1(out_c - c);

                const __fp16 *output0_tm_0 = src + (i * blk_w + j) * out_c + c;  // 8*8 起始地址
                const __fp16 *output0_tm_1 = output0_tm_0 + tiles * out_c * 8;
                const __fp16 *output0_tm_2 = output0_tm_1 + tiles * out_c * 8;
                const __fp16 *output0_tm_3 = output0_tm_2 + tiles * out_c * 8;
                const __fp16 *output0_tm_4 = output0_tm_3 + tiles * out_c * 8;
                const __fp16 *output0_tm_5 = output0_tm_4 + tiles * out_c * 8;
                const __fp16 *output0_tm_6 = output0_tm_5 + tiles * out_c * 8;
                const __fp16 *output0_tm_7 = output0_tm_6 + tiles * out_c * 8;
                __fp16 *output0 = dst + (i * 6 * blk_w * 6 + j * 6) * out_c + c;  // out 6*6 addr

                vfloat16m1_t _bias = bias ? vle16_v_f16m1(bias + c, vl) : vfmv_v_f_f16m1(0.0f, vl);

                for (int m = 0; m < 8; m++) {
                    vfloat16m1_t _r00 = vle16_v_f16m1(output0_tm_0, vl);
                    vfloat16m1_t _r01 = vle16_v_f16m1(output0_tm_1, vl);
                    vfloat16m1_t _r02 = vle16_v_f16m1(output0_tm_2, vl);
                    vfloat16m1_t _r03 = vle16_v_f16m1(output0_tm_3, vl);
                    vfloat16m1_t _r04 = vle16_v_f16m1(output0_tm_4, vl);
                    vfloat16m1_t _r05 = vle16_v_f16m1(output0_tm_5, vl);
                    vfloat16m1_t _r06 = vle16_v_f16m1(output0_tm_6, vl);
                    vfloat16m1_t _r07 = vle16_v_f16m1(output0_tm_7, vl);

                    vfloat16m1_t _tmp024a = vfadd_vv_f16m1(_r01, _r02, vl);
                    vfloat16m1_t _tmp135a = vfsub_vv_f16m1(_r01, _r02, vl);

                    vfloat16m1_t _tmp024b = vfadd_vv_f16m1(_r03, _r04, vl);
                    vfloat16m1_t _tmp135b = vfsub_vv_f16m1(_r03, _r04, vl);

                    vfloat16m1_t _tmp024c = vfadd_vv_f16m1(_r05, _r06, vl);
                    vfloat16m1_t _tmp135c = vfsub_vv_f16m1(_r05, _r06, vl);

                    vfloat16m1_t _tmp0m =
                        vfadd_vv_f16m1(vfadd_vv_f16m1(_r00, _tmp024a, vl),
                                       vfmacc_vf_f16m1(_tmp024b, 32.f, _tmp024c, vl), vl);
                    vfloat16m1_t _tmp2m = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(_tmp024a, 4.f, _tmp024b, vl), 8.f, _tmp024c, vl);
                    vfloat16m1_t _tmp4m = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(_tmp024a, 16.f, _tmp024b, vl), 2.f, _tmp024c, vl);

                    vfloat16m1_t _tmp1m = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(_tmp135a, 2.f, _tmp135b, vl), 16.f, _tmp135c, vl);
                    vfloat16m1_t _tmp3m = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(_tmp135a, 8.f, _tmp135b, vl), 4.f, _tmp135c, vl);
                    vfloat16m1_t _tmp5m =
                        vfadd_vv_f16m1(vfadd_vv_f16m1(_r07, _tmp135a, vl),
                                       vfmacc_vf_f16m1(_tmp135c, 32.f, _tmp135b, vl), vl);

                    vse16_v_f16m1(tmp[0][m], _tmp0m, vl);
                    vse16_v_f16m1(tmp[2][m], _tmp2m, vl);
                    vse16_v_f16m1(tmp[4][m], _tmp4m, vl);
                    vse16_v_f16m1(tmp[1][m], _tmp1m, vl);
                    vse16_v_f16m1(tmp[3][m], _tmp3m, vl);
                    vse16_v_f16m1(tmp[5][m], _tmp5m, vl);

                    output0_tm_0 += tiles * out_c;
                    output0_tm_1 += tiles * out_c;
                    output0_tm_2 += tiles * out_c;
                    output0_tm_3 += tiles * out_c;
                    output0_tm_4 += tiles * out_c;
                    output0_tm_5 += tiles * out_c;
                    output0_tm_6 += tiles * out_c;
                    output0_tm_7 += tiles * out_c;
                }
                for (int m = 0; m < 6; m++) {
                    vfloat16m1_t _tmp00 = vle16_v_f16m1(tmp[m][0], vl);
                    vfloat16m1_t _tmp01 = vle16_v_f16m1(tmp[m][1], vl);
                    vfloat16m1_t _tmp02 = vle16_v_f16m1(tmp[m][2], vl);
                    vfloat16m1_t _tmp03 = vle16_v_f16m1(tmp[m][3], vl);
                    vfloat16m1_t _tmp04 = vle16_v_f16m1(tmp[m][4], vl);
                    vfloat16m1_t _tmp05 = vle16_v_f16m1(tmp[m][5], vl);
                    vfloat16m1_t _tmp06 = vle16_v_f16m1(tmp[m][6], vl);
                    vfloat16m1_t _tmp07 = vle16_v_f16m1(tmp[m][7], vl);

                    vfloat16m1_t _tmp024a = vfadd_vv_f16m1(_tmp01, _tmp02, vl);
                    vfloat16m1_t _tmp135a = vfsub_vv_f16m1(_tmp01, _tmp02, vl);

                    vfloat16m1_t _tmp024b = vfadd_vv_f16m1(_tmp03, _tmp04, vl);
                    vfloat16m1_t _tmp135b = vfsub_vv_f16m1(_tmp03, _tmp04, vl);

                    vfloat16m1_t _tmp024c = vfadd_vv_f16m1(_tmp05, _tmp06, vl);
                    vfloat16m1_t _tmp135c = vfsub_vv_f16m1(_tmp05, _tmp06, vl);

                    vfloat16m1_t _output00 =
                        vfadd_vv_f16m1(vfadd_vv_f16m1(_tmp00, _tmp024a, vl),
                                       vfmacc_vf_f16m1(_tmp024b, 32.f, _tmp024c, vl), vl);
                    vfloat16m1_t _output02 = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(_tmp024a, 4.f, _tmp024b, vl), 8.f, _tmp024c, vl);
                    vfloat16m1_t _output04 = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(_tmp024a, 16.f, _tmp024b, vl), 2.f, _tmp024c, vl);

                    vfloat16m1_t _output01 = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(_tmp135a, 2.f, _tmp135b, vl), 16.f, _tmp135c, vl);
                    vfloat16m1_t _output03 = vfmacc_vf_f16m1(
                        vfmacc_vf_f16m1(_tmp135a, 8.f, _tmp135b, vl), 4.f, _tmp135c, vl);
                    vfloat16m1_t _output05 =
                        vfadd_vv_f16m1(vfadd_vv_f16m1(_tmp07, _tmp135a, vl),
                                       vfmacc_vf_f16m1(_tmp135c, 32.f, _tmp135b, vl), vl);

                    _output00 = vfadd_vv_f16m1(_bias, _output00, vl);
                    _output01 = vfadd_vv_f16m1(_bias, _output01, vl);
                    _output02 = vfadd_vv_f16m1(_bias, _output02, vl);
                    _output03 = vfadd_vv_f16m1(_bias, _output03, vl);
                    _output04 = vfadd_vv_f16m1(_bias, _output04, vl);
                    _output05 = vfadd_vv_f16m1(_bias, _output05, vl);

                    vse16_v_f16m1(output0, _output00, vl);
                    vse16_v_f16m1(output0 + out_c * 2, _output02, vl);
                    vse16_v_f16m1(output0 + out_c * 4, _output04, vl);
                    vse16_v_f16m1(output0 + out_c * 1, _output01, vl);
                    vse16_v_f16m1(output0 + out_c * 3, _output03, vl);
                    vse16_v_f16m1(output0 + out_c * 5, _output05, vl);

                    output0 += blk_w * 6 * out_c;
                }
                c += vl;
            }
        }
    }
}

int shl_rvm_wg_b6f3s1_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)params->conv_extra.kernel_tm->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int pad_left = params->pad_left;
    int pad_top = params->pad_top;

    int batch = input->dim[0];
    int in_h = input->dim[1];
    int in_w = input->dim[2];
    int in_c = input->dim[3];
    int input_size = in_c * in_h * in_w;

    int out_c = kernel->dim[0];
    int out_h = output->dim[1];
    int out_w = output->dim[2];
    int output_size = out_c * out_h * out_w;

    int block_h = (out_h + 5) / 6;
    int block_w = (out_w + 5) / 6;

    // block * 6 for alignment with 6, kernel = 3 * 3, stride = 1, thus input_size + 2
    int padded_in_h = block_h * 6 + 2;
    int padded_in_w = block_w * 6 + 2;
    int padded_in_hw = padded_in_h * padded_in_w;  // element size after padding per channel

    int tiles = block_h * block_w;
    int k_align = ((in_c - 1) & -(csrr_xmlenb() / 2)) + (csrr_xmlenb() / 2);

    for (int b = 0; b < batch; b++) {
        /****************************** pad input *****************************/
        // pad buffer: [blk_h*6+2, blk_w*6+2, in_c]
        __fp16 *input_padd_buf = (__fp16 *)shl_mem_alloc(padded_in_hw * in_c * sizeof(__fp16));
        winograd_pad_input_nhwc_fp16(input_data, input_padd_buf, in_h, in_w, in_c, padded_in_h,
                                     padded_in_w, pad_top, pad_left);
        input_data += input_size;

        /****************************** transform input *****************************/
        // input_tm_buf: [64, tiles, in_c]
        __fp16 *input_tm_buf = (__fp16 *)shl_mem_alloc(64 * tiles * k_align * sizeof(__fp16));
        wg_b6f3s1_trans_input_nhwc_fp16(input_padd_buf, input_tm_buf, padded_in_h, padded_in_w,
                                        in_c, block_h, block_w);
        shl_mem_free(input_padd_buf);

        /****************************** batch gemm *****************************/
        // output_dot_buf: [64, tiles, out_c]
        __fp16 *output_dot_buf = (__fp16 *)shl_mem_alloc(64 * tiles * out_c * sizeof(__fp16));
        wg_bxf3s1_batch_gemm_matrix_fp16(input_tm_buf, kernel_data, output_dot_buf, NULL, k_align,
                                         out_c, tiles, 64);
        shl_mem_free(input_tm_buf);

        /****************************** transform output *****************************/
        // output_tm_buf: [blk_h*6, blk_w*6, out_c]
        __fp16 *output_tm_buf =
            (__fp16 *)shl_mem_alloc(block_h * 6 * block_w * 6 * out_c * sizeof(__fp16));
        wg_b6f3s1_trans_output_nhwc_fp16(output_dot_buf, bias_data, output_tm_buf, block_h, block_w,
                                         out_c);
        shl_mem_free(output_dot_buf);

        /****************************** crop output *****************************/
        // crop the output after transform: cut extra part (right , bottom)
        winograd_crop_output_nhwc_fp16(output_tm_buf, output_data, out_h, out_w, out_c, block_h * 6,
                                       block_w * 6);

        output_data += output_size;
        shl_mem_free(output_tm_buf);
    }
    return CSINN_TRUE;
}
