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

#include "rvv/rvv.h"

/*************************************************************
 * pack4n = vlenb / sizeof(__fp16) * 4
 * src: [N, H, W, C]
 * dst: [N, H, C/pack4n, W, pack4n]
 ************************************************************/
static void reorder_input_pack4n(const __fp16 *src, __fp16 *dst, int32_t *dim)
{
    int batch = dim[0] * dim[1];
    int M = dim[2];
    int N = dim[3];
    for (int b = 0; b < batch; b++) {
        int j = 0;
        while (j < N) {
            int vl = vsetvl_e16m4(N - j);
            for (int i = 0; i < M; i++) {
                const __fp16 *s_ptr = src + i * N + j;
                vfloat16m4_t _src = vle16_v_f16m4(s_ptr, vl);
                vse16_v_f16m4(dst, _src, vl);
                dst += vl;
            }
            j += vl;
        }
        src += M * N;
    }
}

/*************************************************************
 * pack4n = vlenb / sizeof(__fp16) * 4
 * src: [O, H, W, I]
 * dst: [O, H, I/pack4n, W, pack4n]
 ************************************************************/
void shl_rvv_conv3x3s1_direct_reorder_kernel_pack4n_fp16(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params)
{
    int size = csinn_tensor_size(kernel);
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *pa_reorder = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
    reorder_input_pack4n(kernel_data, pa_reorder, kernel->dim);
    memcpy(kernel_data, pa_reorder, size * sizeof(__fp16));
    shl_mem_free(pa_reorder);
}

/*************************************************************
 * pack4n = vlenb / sizeof(__fp16) * 4
 * kernel: [O, H, W, I] -> [O, H, I/pack4n, W, pack4n]
 * input:  [N, H, W, C] -> [N, H, C/pack4n, W, pack4n]
 * output: [N, H, W, C]
 ************************************************************/
int shl_rvv_conv3x3s1_direct_fp16_nhwc(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_h = input->dim[1];
    int32_t in_w = input->dim[2];
    int32_t in_c = input->dim[3];
    int32_t out_h = output->dim[1];
    int32_t out_w = output->dim[2];
    int32_t out_c = output->dim[3];
    int32_t ksize_h = kernel->dim[1];
    int32_t ksize_w = kernel->dim[2];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;
    int32_t pad_top = params->pad_top;
    int32_t pad_down = params->pad_down;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;

    assert(group == 1 && dilation_h == 1 && dilation_w == 1);
    assert(ksize_h == 3 && ksize_w == 3 && stride_h == 1 && stride_w == 1);

    int vl16m4 = vsetvl_e16m4(in_c);
    int buffer_size = out_w * vl16m4 * sizeof(__fp16);
    __fp16 *acc_buffer = (__fp16 *)shl_mem_alloc(buffer_size);

    __fp16 *input_reorder = (__fp16 *)shl_mem_alloc(csinn_tensor_size(input) * sizeof(__fp16));
    reorder_input_pack4n(input_data, input_reorder, input->dim);

    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                const int32_t in_y_origin = (oh * stride_h) - pad_top;
                memset(acc_buffer, 0, buffer_size);
                for (int kh = 0; kh < ksize_h; kh++) {
                    int ih = in_y_origin + kh;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }

                    int ic = 0;
                    __fp16 *kernel_ptr = kernel_data + (oc * ksize_h + kh) * ksize_w * in_c;
                    while (ic < in_c) {
                        int vl = vsetvl_e16m4(in_c - ic);
                        __fp16 *k_ptr = kernel_ptr + ic;
                        vfloat16m4_t _k0 = vle16_v_f16m4(k_ptr, vl);
                        vfloat16m4_t _k1 = vle16_v_f16m4(k_ptr + vl, vl);
                        vfloat16m4_t _k2 = vle16_v_f16m4(k_ptr + vl * 2, vl);
                        vfloat16m4_t _in0;
                        vfloat16m4_t _in1;
                        vfloat16m4_t _in2;
                        __fp16 *input_ptr = input_reorder + ((b * in_h + ih) * in_c + ic) * in_w;
                        for (int ow = 0; ow < out_w; ow++) {
                            const int32_t in_x_origin = (ow * stride_w) - pad_left;
                            int iw0 = in_x_origin;
                            int iw1 = in_x_origin + 1;
                            int iw2 = in_x_origin + 2;

                            if (ow == 0 && iw1 >= 0) {
                                __fp16 *in1_ptr = input_ptr + iw1 * vl;
                                _in1 = vle16_v_f16m4(in1_ptr, vl);
                                if (iw0 >= 0) {
                                    __fp16 *in0_ptr = input_ptr + iw0 * vl;
                                    _in0 = vle16_v_f16m4(in0_ptr, vl);
                                }
                            }

                            // without pad
                            if (iw0 >= 0 && iw2 < in_w) {
                                __fp16 *acc_ptr = acc_buffer + ow * vl16m4;
                                vfloat16m4_t _acc = vle16_v_f16m4(acc_ptr, vl);
                                __fp16 *in2_ptr = input_ptr + iw2 * vl;
                                if (ow % 3 == 0) {
                                    _in2 = vle16_v_f16m4(in2_ptr, vl);
                                    _acc = vfmacc_vv_f16m4(_acc, _in0, _k0, vl);
                                    _acc = vfmacc_vv_f16m4(_acc, _in1, _k1, vl);
                                    _acc = vfmacc_vv_f16m4(_acc, _in2, _k2, vl);
                                } else if (ow % 3 == 1) {
                                    _in0 = vle16_v_f16m4(in2_ptr, vl);
                                    _acc = vfmacc_vv_f16m4(_acc, _in1, _k0, vl);
                                    _acc = vfmacc_vv_f16m4(_acc, _in2, _k1, vl);
                                    _acc = vfmacc_vv_f16m4(_acc, _in0, _k2, vl);
                                } else if (ow % 3 == 2) {
                                    _in1 = vle16_v_f16m4(in2_ptr, vl);
                                    _acc = vfmacc_vv_f16m4(_acc, _in2, _k0, vl);
                                    _acc = vfmacc_vv_f16m4(_acc, _in0, _k1, vl);
                                    _acc = vfmacc_vv_f16m4(_acc, _in1, _k2, vl);
                                }
                                vse16_v_f16m4(acc_ptr, _acc, vl);
                            }

                            // with pad
                            else {
                                __fp16 *acc_ptr = acc_buffer + ow * vl16m4;
                                vfloat16m4_t _acc = vle16_v_f16m4(acc_ptr, vl);
                                if (iw0 >= 0 && iw0 < in_w) {
                                    if (ow % 3 == 0) {
                                        _acc = vfmacc_vv_f16m4(_acc, _in0, _k0, vl);
                                    } else if (ow % 3 == 1) {
                                        _acc = vfmacc_vv_f16m4(_acc, _in1, _k0, vl);
                                    } else if (ow % 3 == 2) {
                                        _acc = vfmacc_vv_f16m4(_acc, _in2, _k0, vl);
                                    }
                                }
                                if (iw1 >= 0 && iw1 < in_w) {
                                    if (ow % 3 == 0) {
                                        _acc = vfmacc_vv_f16m4(_acc, _in1, _k1, vl);
                                    } else if (ow % 3 == 1) {
                                        _acc = vfmacc_vv_f16m4(_acc, _in2, _k1, vl);
                                    } else if (ow % 3 == 2) {
                                        _acc = vfmacc_vv_f16m4(_acc, _in0, _k1, vl);
                                    }
                                }
                                if (iw2 >= 0 && iw2 < in_w) {
                                    __fp16 *in2_ptr = input_ptr + iw2 * vl;
                                    if (ow % 3 == 0) {
                                        _in2 = vle16_v_f16m4(in2_ptr, vl);
                                        _acc = vfmacc_vv_f16m4(_acc, _in2, _k2, vl);
                                    } else if (ow % 3 == 1) {
                                        _in0 = vle16_v_f16m4(in2_ptr, vl);
                                        _acc = vfmacc_vv_f16m4(_acc, _in0, _k2, vl);
                                    } else if (ow % 3 == 2) {
                                        _in1 = vle16_v_f16m4(in2_ptr, vl);
                                        _acc = vfmacc_vv_f16m4(_acc, _in1, _k2, vl);
                                    }
                                }
                                vse16_v_f16m4(acc_ptr, _acc, vl);
                            }
                        }
                        ic += vl;
                    }
                }

                for (int ow = 0; ow < out_w; ow++) {
                    __fp16 *acc_ptr = acc_buffer + ow * vl16m4;
                    vfloat16m1_t _sum = vfmv_v_f_f16m1(bias_data[oc], 1);
                    vfloat16m4_t _acc = vle16_v_f16m4(acc_ptr, vl16m4);
                    _sum = vfredusum_vs_f16m4_f16m1(vundefined_f16m1(), _acc, _sum, vl16m4);
                    output_data[((b * out_h + oh) * out_w + ow) * out_c + oc] =
                        vfmv_f_s_f16m1_f16(_sum);
                }
            }
        }
    }
    shl_mem_free(acc_buffer);
    shl_mem_free(input_reorder);

    return CSINN_TRUE;
}
