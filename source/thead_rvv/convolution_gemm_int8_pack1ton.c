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

#include "shl_thead_rvv.h"

/*************************************************************
 * packn = vlenb / sizeof(int8_t) / 2
 * maxk = ksize_h * ksize_w
 * constrain: out_c % packn = 0 and in_ch % packn can != 0
 * layout: [out_c/packna, in_c/packnb*maxk*packnb + maxk*in_c%packnb, packna] -- dot
 * layout: [out_c/pack2n, in_c/packn*maxk*packn + maxk*in_c%packn, pack2n]
 *         [out_c/packna, in_c/packnb*maxk*packnb + maxk*in_c%packnb, packna] -- without dot
 ************************************************************/
static void im2col_gemm_reorder_kernel_pack1ton_per_group_int8(int8_t *src, int8_t *dst, int out_c,
                                                               int in_c, int maxk)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
#ifdef SHL_USE_DOT_INT8
    const int vl = vsetvl_e8mf2(packn);
    int in_c4 = ((in_c - 1) & -4) + 4;
    for (int oc = 0; oc + packn - 1 < out_c; oc += packn) {
        int8_t *k0 = src + oc * in_c * maxk;
        int8_t *g0 = dst + oc * in_c4 * maxk;

        int ic = 0;
        for (; ic + packn - 1 < in_c; ic += packn) {
            for (int k = 0; k < maxk; k++) {
                int8_t *g1 = g0 + (ic * maxk) * packn + k * packn * packn;

                for (int p = 0; p < packn / 4; p++) {
                    int8_t *g2 = g1 + p * 4 * packn;
                    for (int i = 0; i < 4; i++) {
                        vint8mf2_t _tmp = vlse8_v_i8mf2(k0 + (ic + p * 4 + i) * maxk + k,
                                                        in_c * maxk * sizeof(int8_t), vl);
                        vsse8_v_i8mf2(g2, 4 * sizeof(int8_t), _tmp, vl);
                        g2++;
                    }
                }
            }
        }
        if (ic < in_c) {
            int tail_c = in_c & (packn - 1);
            int tail_c4 = in_c & 3;
            for (int k = 0; k < maxk; k++) {
                int8_t *g1 = g0 + (ic * maxk) * packn + k * packn * (in_c4 - ic);

                int p = 0;
                for (; p + 3 < tail_c; p += 4) {
                    int8_t *g2 = g1 + p * packn;
                    for (int i = 0; i < 4; i++) {
                        vint8mf2_t _tmp = vlse8_v_i8mf2(k0 + (ic + p + i) * maxk + k,
                                                        in_c * maxk * sizeof(int8_t), vl);
                        vsse8_v_i8mf2(g2, 4 * sizeof(int8_t), _tmp, vl);
                        g2++;
                    }
                }
                if (p < tail_c) {
                    int8_t *g2 = g1 + p * packn;
                    for (int i = 0; i < tail_c4; i++) {
                        vint8mf2_t _tmp = vlse8_v_i8mf2(k0 + (ic + p + i) * maxk + k,
                                                        in_c * maxk * sizeof(int8_t), vl);
                        vsse8_v_i8mf2(g2, 4 * sizeof(int8_t), _tmp, vl);
                        g2++;
                    }
                }
            }
        }
    }
#else
    const int pack2n = packn * 2;
    int vl = vsetvl_e8m1(pack2n);
    int oc = 0;
    // [out_c/pack2n, in_c/packn*maxk*packn + maxk*in_c%packn, pack2n]
    for (; oc + pack2n - 1 < out_c; oc += pack2n) {
        int8_t *k0 = src + oc * in_c * maxk;
        int8_t *g0 = dst + oc * in_c * maxk;

        int ic = 0;
        for (; ic + packn - 1 < in_c; ic += packn) {
            for (int k = 0; k < maxk; k++) {
                for (int p = 0; p < packn; p++) {
                    vint8m1_t _tmp =
                        vlse8_v_i8m1(k0 + ((ic + p) * maxk + k), in_c * maxk * sizeof(int8_t), vl);
                    vse8_v_i8m1(g0, _tmp, vl);
                    g0 += vl;
                }
            }
        }
        if (ic < in_c) {
            int tail_c = in_c & (packn - 1);
            for (int k = 0; k < maxk; k++) {
                for (int p = 0; p < tail_c; p++) {
                    vint8m1_t _tmp =
                        vlse8_v_i8m1(k0 + ((ic + p) * maxk + k), in_c * maxk * sizeof(int8_t), vl);
                    vse8_v_i8m1(g0, _tmp, vl);
                    g0 += vl;
                }
            }
        }
    }
    vl = vsetvl_e8m1(packn);
    // [out_c/packn, in_c/packnb*maxk*packnb + maxk*in_c%packnb, packn]
    for (; oc + packn - 1 < out_c; oc += packn) {
        int8_t *k0 = src + oc * in_c * maxk;
        int8_t *g0 = dst + oc * in_c * maxk;

        int ic = 0;
        for (; ic + packn - 1 < in_c; ic += packn) {
            for (int k = 0; k < maxk; k++) {
                for (int p = 0; p < packn; p++) {
                    vint8m1_t _tmp =
                        vlse8_v_i8m1(k0 + ((ic + p) * maxk + k), in_c * maxk * sizeof(int8_t), vl);
                    vse8_v_i8m1(g0, _tmp, vl);
                    g0 += vl;
                }
            }
        }
        if (ic < in_c) {
            int tail_c = in_c & (packn - 1);
            for (int k = 0; k < maxk; k++) {
                for (int p = 0; p < tail_c; p++) {
                    vint8m1_t _tmp =
                        vlse8_v_i8m1(k0 + ((ic + p) * maxk + k), in_c * maxk * sizeof(int8_t), vl);
                    vse8_v_i8m1(g0, _tmp, vl);
                    g0 += vl;
                }
            }
        }
    }
#endif  // SHL_USE_DOT_INT8
}

void shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_int8(struct csinn_tensor *kernel,
                                                           struct csinn_conv2d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    int group = params->group;

    int out_c = kernel->dim[0];
    int out_cp = out_c / group;  // per-group out channel
    int in_c = kernel->dim[1];
    int maxk = kernel->dim[2] * kernel->dim[3];

#ifdef SHL_USE_DOT_INT8
    int in_c4 = ((in_c - 1) & -4) + 4;  // align 4 for input_channel
    params->conv_extra.kernel_tm->data =
        (int8_t *)shl_mem_alloc(out_c * in_c4 * maxk * sizeof(int8_t));
    int8_t *pa_reorder = (int8_t *)params->conv_extra.kernel_tm->data;

    for (int g = 0; g < group; g++) {
        int8_t *ker_ptr = kernel_data + g * out_cp * in_c * maxk;
        int8_t *ker_tm_ptr = pa_reorder + g * out_cp * in_c4 * maxk;
        im2col_gemm_reorder_kernel_pack1ton_per_group_int8(ker_ptr, ker_tm_ptr, out_cp, in_c, maxk);
    }
#else   // in_channel 无需按照 4 对齐
    params->conv_extra.kernel_tm->data =
        (int8_t *)shl_mem_alloc(out_c * in_c * maxk * sizeof(int8_t));
    int8_t *pa_reorder = (int8_t *)params->conv_extra.kernel_tm->data;
    for (int g = 0; g < group; g++) {
        int8_t *ker_ptr = kernel_data + g * out_cp * in_c * maxk;
        int8_t *ker_tm_ptr = pa_reorder + g * out_cp * in_c * maxk;
        im2col_gemm_reorder_kernel_pack1ton_per_group_int8(ker_ptr, ker_tm_ptr, out_cp, in_c, maxk);
    }
#endif  // SHL_USE_DOT_INT8
}

int shl_rvv_conv_im2col_gemm_pack1ton_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)params->conv_extra.kernel_tm->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t out_c = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];
    int32_t ksize_h = kernel->dim[2];
    int32_t ksize_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    int32_t m = out_c / group;
    int32_t in_cp = in_c / group;
    int32_t maxk = ksize_h * ksize_w;
    int32_t n = out_h * out_w;

    int32_t *multiplier = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));

    for (int i = 0; i < batch; i++) {
        for (int g = 0, j = 0; g < group; g++) {
            // padding
            int padded_in_hw = (in_h + params->pad_top + params->pad_down) *
                               (in_w + params->pad_left + params->pad_right);
            int8_t *input_pad_buf = (int8_t *)shl_mem_alloc(in_cp * padded_in_hw * sizeof(int8_t));
            shl_rvv_pad_input_pack1ton_int8(input_data, input_pad_buf, in_cp, in_h, in_w,
                                            (in_h + params->pad_top + params->pad_down),
                                            (in_w + params->pad_left + params->pad_right),
                                            params->pad_top, params->pad_left,
                                            input->qinfo->zero_point);

            if (kernel->quant_channel > 1) {
                for (int c = 0; c < m; c++, j++) {
                    multiplier[c] = kernel->qinfo[j].multiplier;
                    shift[c] = kernel->qinfo[j].shift;
                }
            } else if (kernel->quant_channel == 1) {
                for (int c = 0; c < m; c++) {
                    multiplier[c] = kernel->qinfo[0].multiplier;
                    shift[c] = kernel->qinfo[0].shift;
                }
            }
            int32_t *bias_ptr = bias_data + g * m;
            // im2col
            const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
            int vl = vsetvl_e8m1(packn);

#ifdef SHL_USE_DOT_INT8
            // im2col
            int in_cp4 = ((in_cp - 1) & -4) + 4;
            // [in_cp4/packn, maxk, out_h, out_w, packn] + [maxk, out_h, out_w, in_cp4%packn]
            int8_t *im2col_buf = (int8_t *)shl_mem_alloc(in_cp4 * maxk * n * sizeof(int8_t));
            const int tailstep =
                ((in_w + params->pad_left + params->pad_right) * stride_h - out_w * stride_w);

            const int8_t *img0 = input_pad_buf;
            int8_t *dst_ptr = im2col_buf;

            int loop_c = in_cp;
            while (loop_c > 0) {
                vl = vsetvl_e8mf2(loop_c);
                int vl4 = ((vl - 1) & -4) + 4;
                for (int a = 0; a < ksize_h; a++) {
                    for (int b = 0; b < ksize_w; b++) {
                        const int8_t *img1 =
                            img0 + a * (in_w + params->pad_left + params->pad_right) * vl + b * vl;

                        for (int p = 0; p < out_h; p++) {
                            for (int q = 0; q < out_w; q++) {
                                vint8mf2_t _tmp = vle8_v_i8mf2(img1, vl);
                                img1 += stride_w * vl;
                                vse8_v_i8mf2(dst_ptr, _tmp, vl);
                                dst_ptr += vl4;  // XXX: dst align 4
                            }
                            img1 += tailstep * vl;
                        }
                    }
                }
                img0 += padded_in_hw * vl;
                // dst_ptr += maxk * out_h * out_w * vl;
                loop_c -= vl;
            }
            shl_mem_free(input_pad_buf);
            // reorder
            int8_t *reorder_buf = (int8_t *)shl_mem_alloc(in_cp4 * maxk * n * sizeof(int8_t));
            shl_rvv_reorder_input_z12_pack1ton_int8_dot(im2col_buf, reorder_buf, in_cp4, maxk, n,
                                                        n);
            shl_mem_free(im2col_buf);
            int8_t *ker_ptr = kernel_data + g * m * maxk * in_cp4;
            // gemm
            shl_rvv_ncxhwx_gemm_12xpackn_int8_dot(output_data, ker_ptr, reorder_buf, bias_ptr, m,
                                                  in_cp4 * maxk, n, n, output->qinfo->zero_point,
                                                  multiplier, shift);
#else
            // im2col
            // [in_c/packn, maxk, out_h, out_w, packn] + [maxk, out_h, out_w, in_c%packn]
            int8_t *im2col_buf = (int8_t *)shl_mem_alloc(in_cp * maxk * n * sizeof(int8_t));
            const int tailstep =
                ((in_w + params->pad_left + params->pad_right) * stride_h - out_w * stride_w);

            const int8_t *img0 = input_pad_buf;
            int8_t *dst_ptr = im2col_buf;

            int loop_c = in_cp;
            while (loop_c > 0) {
                vl = vsetvl_e8m1(loop_c > packn ? packn : loop_c);
                for (int a = 0; a < ksize_h; a++) {
                    for (int b = 0; b < ksize_w; b++) {
                        const int8_t *img1 =
                            img0 + a * (in_w + params->pad_left + params->pad_right) * vl + b * vl;

                        for (int p = 0; p < out_h; p++) {
                            for (int q = 0; q < out_w; q++) {
                                vint8m1_t _tmp = vle8_v_i8m1(img1, vl);
                                img1 += stride_w * vl;
                                vse8_v_i8m1(dst_ptr, _tmp, vl);
                                dst_ptr += vl;
                            }
                            img1 += tailstep * vl;
                        }
                    }
                }
                img0 += padded_in_hw * vl;
                loop_c -= vl;
            }
            shl_mem_free(input_pad_buf);

            // reorder(pack)
            int8_t *reorder_buf = (int8_t *)shl_mem_alloc(in_cp * maxk * n * sizeof(int8_t));
            shl_rvv_reorder_input_z4_pack1ton_int8(im2col_buf, reorder_buf, in_cp, maxk, n, n);
            shl_mem_free(im2col_buf);
            int8_t *ker_ptr = kernel_data + g * m * maxk * in_cp;
            // gemm
            shl_rvv_ncxhwx_gemm_4xpack2n_int8(output_data, ker_ptr, reorder_buf, bias_ptr, m,
                                              in_cp * maxk, n, n, output->qinfo->zero_point,
                                              multiplier, shift);
#endif  // SHL_USE_DOT_INT8

            shl_mem_free(reorder_buf);
            input_data += in_cp * in_h * in_w;
            output_data += m * n;
        }
    }
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    return CSINN_TRUE;
}
