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

#include "rvv/rvv.h"

/*************************************************************
 * packn = vlenb / sizeof(int8_t) / 2
 * maxk = ksize_h * ksize_w
 * constrain: out_c % packn != 0 and in_ch % packn = 0
 * layout: [out_c/packna, in_c/packnb, maxk, packnb/4, packna, 4]
 *         [out_c/tail, in_c/packnb, maxk, packnb/4, tail, 4]   -- dot
 * layout: [out_c/pack2n, in_c/packn, maxk, packn, pack2n]
 *         [out_c/packna, in_c/packnb, maxk, packnb, packna]
 *         [out_c/tail, in_c/packnb, maxk, packnb, tail]  -- without dot
 ************************************************************/
static void im2col_gemm_reorder_kernel_packnto1_per_group_int8(int8_t *src, int8_t *dst, int out_c,
                                                               int in_c, int maxk)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
#ifdef SHL_USE_DOT_INT8
    int vl = vsetvl_e8mf2(packn);
    // [out_c/packna, in_c/packnb, maxk, packnb/4, packna, 4b]
    int oc = 0;
    for (; oc + packn - 1 < out_c; oc += packn) {
        int8_t *k0 = src + oc * in_c * maxk;
        int8_t *g0 = dst + oc * in_c / packn * maxk * packn / 4 * 4;

        for (int ic = 0; ic + packn - 1 < in_c; ic += packn) {
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
    }
    // [out_c/tail, in_c/packnb, maxk, packnb/4, tail, 4]
    if (oc < out_c) {
        vl = vsetvl_e8mf2(out_c - oc);
        int8_t *k0 = src + oc * in_c * maxk;
        int8_t *g0 = dst + oc * in_c / packn * maxk * packn / 4 * 4;

        for (int ic = 0; ic + packn - 1 < in_c; ic += packn) {
            for (int k = 0; k < maxk; k++) {
                int8_t *g1 = g0 + (ic * maxk) * vl + k * packn * vl;

                for (int p = 0; p < packn / 4; p++) {
                    int8_t *g2 = g1 + p * 4 * vl;
                    for (int i = 0; i < 4; i++) {
                        vint8mf2_t _tmp = vlse8_v_i8mf2(k0 + (ic + p * 4 + i) * maxk + k,
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
    // [out_c/pack2n, in_c/packn, maxk, packn, pack2n]
    for (; oc + pack2n - 1 < out_c; oc += pack2n) {
        int8_t *k0 = src + oc * in_c * maxk;
        int8_t *g0 = dst + oc * in_c / packn * maxk * packn;

        for (int ic = 0; ic + packn - 1 < in_c; ic += packn) {
            for (int k = 0; k < maxk; k++) {
                for (int p = 0; p < packn; p++) {
                    vint8m1_t _tmp =
                        vlse8_v_i8m1(k0 + ((ic + p) * maxk + k), in_c * maxk * sizeof(int8_t), vl);
                    vse8_v_i8m1(g0, _tmp, vl);
                    g0 += vl;
                }
            }
        }
    }
    vl = vsetvl_e8m1(packn);
    // [out_c/packn, in_c/packn, maxk, packn, packn]
    for (; oc + packn - 1 < out_c; oc += packn) {
        int8_t *k0 = src + oc * in_c * maxk;
        int8_t *g0 = dst + oc * in_c / packn * maxk * packn;

        for (int ic = 0; ic + packn - 1 < in_c; ic += packn) {
            for (int k = 0; k < maxk; k++) {
                for (int p = 0; p < packn; p++) {
                    vint8m1_t _tmp =
                        vlse8_v_i8m1(k0 + ((ic + p) * maxk + k), in_c * maxk * sizeof(int8_t), vl);
                    vse8_v_i8m1(g0, _tmp, vl);
                    g0 += vl;
                }
            }
        }
    }
    // [out_c/tail, in_c/packnb, maxk, packnb, tail]
    if (oc < out_c) {
        vl = vsetvl_e8m1(out_c - oc);
        int8_t *k0 = src + oc * in_c * maxk;
        int8_t *g0 = dst + oc * in_c / packn * maxk * packn;

        for (int ic = 0; ic + packn - 1 < in_c; ic += packn) {
            for (int k = 0; k < maxk; k++) {
                for (int p = 0; p < packn; p++) {
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

void shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_int8(struct csinn_tensor *kernel,
                                                           struct csinn_conv2d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    int group = params->group;

    int out_c = kernel->dim[0];
    int out_cp = out_c / group;  // per-group out channel
    int in_c = kernel->dim[1];
    int maxk = kernel->dim[2] * kernel->dim[3];

    csinn_tensor_copy(params->conv_extra.kernel_tm, kernel);
    params->conv_extra.kernel_tm->data =
        (int8_t *)shl_mem_alloc(out_c * in_c * maxk * sizeof(int8_t));

    for (int g = 0; g < group; g++) {
        int8_t *ker_ptr = kernel_data + g * out_cp * in_c * maxk;
        int8_t *ker_tm_ptr = params->conv_extra.kernel_tm->data + g * out_cp * in_c * maxk;
        im2col_gemm_reorder_kernel_packnto1_per_group_int8(ker_ptr, ker_tm_ptr, out_cp, in_c, maxk);
    }
    kernel->data = NULL;
    // FIXME: free params->conv_extra.kernel_tm->data
    // memcpy(kernel_data, pa_reorder, group * m * k * sizeof(__fp16));
    // shl_mem_free(pa_reorder);
}

int shl_rvv_common_conv_gemm_packnto1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params,
                                           void (*reorder_input)(int8_t *, int8_t *, int, int, int),
                                           void (*gemm)(int8_t *, const int8_t *, const int8_t *,
                                                        int32_t *, int, int, int, int32_t,
                                                        int32_t *, int32_t *))
{
    if (input->layout == CSINN_LAYOUT_NCHW) {
        shl_rvv_tensor_ndarray_to_nc1xc0_replace_int8(input);
    }
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)params->conv_extra.kernel_tm->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1] * input->dim[4];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t out_c = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];
    int32_t ksize_h = kernel->dim[2];
    int32_t ksize_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;

    int32_t m = out_c / group;
    int32_t in_cp = in_c / group;
    int32_t maxk = ksize_h * ksize_w;
    int32_t n = out_h * out_w;

    int8_t *output_ncxhwx = (int8_t *)shl_mem_alloc(m * n * sizeof(int8_t));

    int32_t *multiplier = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));

    for (int i = 0; i < batch; i++) {
        for (int g = 0, j = 0; g < group; g++) {
            // paddding
            int padded_in_h = in_h + params->pad_top + params->pad_down;
            int padded_in_w = in_w + params->pad_left + params->pad_right;
            int padded_in_hw = padded_in_h * padded_in_w;
            int8_t *input_pad_buf = (int8_t *)shl_mem_alloc(in_cp * padded_in_hw * sizeof(int8_t));
            shl_rvv_pad_input_packn_int8(input_data, input_pad_buf, in_cp, in_h, in_w, padded_in_h,
                                         padded_in_w, params->pad_top, params->pad_left,
                                         input->qinfo->zero_point);

            // im2col
            const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
#ifdef RVV_1_0_0
            const int vl = vsetvl_e8mf2(packn);
#elif defined RVV_0_7_1
            const int vl = vsetvl_e8m1(packn);
#endif

            // [in_c/packn, maxk, out_h, out_w, packn]
            int8_t *im2col_buf = (int8_t *)shl_mem_alloc(in_cp / packn * maxk * out_h * out_w *
                                                         packn * sizeof(int8_t));
            const int tailstep = (padded_in_w * stride_h - out_w * stride_w) * packn;

            for (int c = 0; c + packn - 1 < in_cp; c += packn) {
                const int8_t *img0 = input_pad_buf + c * padded_in_hw;
                int8_t *dst_ptr = im2col_buf + c * maxk * out_h * out_w;

                for (int a = 0; a < ksize_h; a++) {
                    for (int b = 0; b < ksize_w; b++) {
                        const int8_t *img1 =
                            img0 + a * dilation_h * padded_in_w * packn + b * dilation_w * packn;

                        for (int p = 0; p < out_h; p++) {
                            for (int q = 0; q < out_w; q++) {
#ifdef RVV_1_0_0
                                vint8mf2_t _tmp = vle8_v_i8mf2(img1, vl);
                                img1 += stride_w * packn;
                                vse8_v_i8mf2(dst_ptr, _tmp, vl);
                                dst_ptr += packn;
#elif defined RVV_0_7_1
                                vint8m1_t _tmp = vle8_v_i8m1(img1, vl);
                                img1 += stride_w * packn;
                                vse8_v_i8m1(dst_ptr, _tmp, vl);
                                dst_ptr += packn;
#endif
                            }
                            img1 += tailstep;
                        }
                    }
                }
            }
            shl_mem_free(input_pad_buf);

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

            int8_t *ker_ptr = kernel_data + g * m * maxk * in_cp;
            int32_t *bias_ptr = bias_data + g * m;  // bias_data != NULL with fusing zp to bias

            int8_t *reorder_buf = (int8_t *)shl_mem_alloc(in_cp * maxk * n * sizeof(int8_t));

            reorder_input(im2col_buf, reorder_buf, in_cp * maxk, n, n);
            shl_mem_free(im2col_buf);
            gemm(output_ncxhwx, ker_ptr, reorder_buf, bias_ptr, m, in_cp * maxk, n,
                 output->qinfo->zero_point, multiplier, shift);

            shl_rvv_reorder_input_packnto1_int8(output_ncxhwx, output_data, m, out_h, out_w);
            shl_mem_free(reorder_buf);

            input_data += in_cp * in_h * in_w;
            output_data += m * n;
        }
    }
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    shl_mem_free(output_ncxhwx);
    return CSINN_TRUE;
}

int shl_rvv_conv_im2col_gemm_packnto1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params)
{
#ifdef SHL_USE_DOT_INT8
    return shl_rvv_common_conv_gemm_packnto1_int8(input, output, kernel, bias, params,
                                                  shl_rvv_reorder_input_z12_packn_int8_dot,
                                                  shl_rvv_ncxhwx_gemm_12xpackn_int8_dot);
#else
    return shl_rvv_common_conv_gemm_packnto1_int8(input, output, kernel, bias, params,
                                                  shl_rvv_reorder_input_z4_packn_int8,
                                                  shl_rvv_ncxhwx_gemm_4xpack2n_int8);
#endif  // SHL_USE_DOT_INT8
}
