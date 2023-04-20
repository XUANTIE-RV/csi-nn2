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

/* SHL version 2.1.x */

#include "shl_thead_rvv.h"

/*************************************************************
    note: VLEN = 128/256 ...
*************************************************************/
static void winograd_pad_input_packn_fp32(const float *input, float *input_padded, int inc, int inh,
                                          int inw, int padded_h, int padded_w, int pad_top,
                                          int pad_left)
{
    shl_rvv_pad_input_packn_fp32(input, input_padded, inc, inh, inw, padded_h, padded_w, pad_top,
                                 pad_left);
}

static void winograd_crop_output_packn_fp32(const float *output_trans, float *output, int out_c,
                                            int out_h, int out_w, int wino_h, int wino_w)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);

    const int out_size = out_h * out_w;  // per-channel size
    const int crop_size = wino_h * wino_w;

    int c = 0;
    for (; c + packn - 1 < out_c; c += packn) {
        float *out_tm_ptr = (float *)output_trans + c * crop_size;
        float *out_ptr = output + c * out_size;

        for (int h = 0; h < out_h; h++) {
            float *crop_ptr = out_tm_ptr + h * wino_w * packn;
            for (int w = 0; w < out_w; w++) {
                vfloat32m1_t _tmp = vle32_v_f32m1(crop_ptr, vl);
                crop_ptr += packn;
                vse32_v_f32m1(out_ptr, _tmp, vl);
                out_ptr += packn;
            }
        }
    }
}

static inline void wg_b4f3s1_trans_input_packn_fp32(const float *src, float *dst, int ch, int h,
                                                    int w, int blk_h, int blk_w)
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
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);
    int tiles = blk_h * blk_w;
    for (int q = 0; q + packn - 1 < ch; q += packn) {
        const float *img0 = src + q * h * w;    // after padding - q channel
        float *img0_tm = dst + q * 36 * tiles;  // transform and interleave - q channel

        float tmp[6][6][packn];

        for (int i = 0; i < blk_h; i++) {
            for (int j = 0; j < blk_w; j++) {
                // pad_buf 6*6 block start addr
                const float *r0 = img0 + (i * w * 4 + j * 4) * packn;
                // input_tm1 6*6 block start addr
                float *r0_tm = img0_tm + (i * blk_w + j) * packn;

                for (int m = 0; m < 6; m++) {
                    vfloat32m1_t _r00 = vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r01 = vle32_v_f32m1(r0 + packn * 1, vl);
                    vfloat32m1_t _r02 = vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _r03 = vle32_v_f32m1(r0 + packn * 3, vl);
                    vfloat32m1_t _r04 = vle32_v_f32m1(r0 + packn * 4, vl);
                    vfloat32m1_t _r05 = vle32_v_f32m1(r0 + packn * 5, vl);

                    vfloat32m1_t _tmp0m =
                        vfmacc_vf_f32m1(vfmacc_vf_f32m1(_r04, 4.f, _r00, vl), -5.f, _r02, vl);
                    vfloat32m1_t _tmp1m = vfmacc_vf_f32m1(vfadd_vv_f32m1(_r04, _r03, vl), -4.f,
                                                          vfadd_vv_f32m1(_r01, _r02, vl), vl);
                    vfloat32m1_t _tmp2m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r04, _r03, vl), 4.f,
                                                          vfsub_vv_f32m1(_r01, _r02, vl), vl);
                    vfloat32m1_t _tmp3m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r04, _r02, vl), -2.f,
                                                          vfsub_vv_f32m1(_r01, _r03, vl), vl);
                    vfloat32m1_t _tmp4m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r04, _r02, vl), 2.f,
                                                          vfsub_vv_f32m1(_r01, _r03, vl), vl);
                    vfloat32m1_t _tmp5m =
                        vfmacc_vf_f32m1(vfmacc_vf_f32m1(_r05, 4.f, _r01, vl), -5.f, _r03, vl);

                    vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                    vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                    vse32_v_f32m1(tmp[4][m], _tmp4m, vl);
                    vse32_v_f32m1(tmp[5][m], _tmp5m, vl);
                    r0 += w * packn;
                }

                for (int m = 0; m < 6; m++) {
                    float *r0_tm0 = r0_tm;
                    float *r0_tm1 = r0_tm0 + tiles * packn;
                    float *r0_tm2 = r0_tm1 + tiles * packn;
                    float *r0_tm3 = r0_tm2 + tiles * packn;
                    float *r0_tm4 = r0_tm3 + tiles * packn;
                    float *r0_tm5 = r0_tm4 + tiles * packn;

                    vfloat32m1_t _tmp00 = vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _tmp01 = vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _tmp02 = vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _tmp03 = vle32_v_f32m1(tmp[m][3], vl);
                    vfloat32m1_t _tmp04 = vle32_v_f32m1(tmp[m][4], vl);
                    vfloat32m1_t _tmp05 = vle32_v_f32m1(tmp[m][5], vl);

                    vfloat32m1_t _r0tm0 =
                        vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp04, 4.f, _tmp00, vl), -5.f, _tmp02, vl);
                    vfloat32m1_t _r0tm1 = vfmacc_vf_f32m1(vfadd_vv_f32m1(_tmp04, _tmp03, vl), -4.f,
                                                          vfadd_vv_f32m1(_tmp01, _tmp02, vl), vl);
                    vfloat32m1_t _r0tm2 = vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp04, _tmp03, vl), 4.f,
                                                          vfsub_vv_f32m1(_tmp01, _tmp02, vl), vl);
                    vfloat32m1_t _r0tm3 = vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp04, _tmp02, vl), -2.f,
                                                          vfsub_vv_f32m1(_tmp01, _tmp03, vl), vl);
                    vfloat32m1_t _r0tm4 = vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp04, _tmp02, vl), 2.f,
                                                          vfsub_vv_f32m1(_tmp01, _tmp03, vl), vl);
                    vfloat32m1_t _r0tm5 =
                        vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp05, 4.f, _tmp01, vl), -5.f, _tmp03, vl);

                    vse32_v_f32m1(r0_tm0, _r0tm0, vl);
                    vse32_v_f32m1(r0_tm1, _r0tm1, vl);
                    vse32_v_f32m1(r0_tm2, _r0tm2, vl);
                    vse32_v_f32m1(r0_tm3, _r0tm3, vl);
                    vse32_v_f32m1(r0_tm4, _r0tm4, vl);
                    vse32_v_f32m1(r0_tm5, _r0tm5, vl);
                    r0_tm += tiles * packn * 6;
                }
            }
        }
    }
}

static inline void wg_b4f3s1_trans_output_packn_fp32(const float *src, const float *bias,
                                                     float *dst, int ch, int blk_h, int blk_w)
{
    /* output transform matrix
    AT = {
        { 1  1  1   1  1   0 },
        { 0  1  -1  2  -2  0 },
        { 0  1  1   4  4   0 },
        { 0  1  -1  8  -8  1 }
    };
    */
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);
    int tiles = blk_h * blk_w;
    for (int p = 0; p + packn - 1 < ch; p += packn) {
        const float *out0_tm = src + p * 36 * tiles;    // 输出转换前/dot后 第p个channel
        float *out0 = dst + p * 4 * blk_h * 4 * blk_w;  // 转换后输出 第p个channel

        float tmp[4][6][packn];

        vfloat32m1_t _bias = bias ? vle32_v_f32m1(bias + p, vl) : vfmv_v_f_f32m1(0.0f, vl);

        for (int i = 0; i < blk_h; i++) {
            for (int j = 0; j < blk_w; j++) {
                const float *output0_tm_0 = out0_tm + (i * blk_w + j) * packn;  // 6*6 起始地址
                const float *output0_tm_1 = output0_tm_0 + tiles * packn * 1;
                const float *output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const float *output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                const float *output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                const float *output0_tm_5 = output0_tm_0 + tiles * packn * 5;

                float *output0 = out0 + (i * blk_w * 4 * 4 + j * 4) * packn;  // out 4*4 addr

                for (int m = 0; m < 6; m++) {
                    vfloat32m1_t _r00 = vle32_v_f32m1(output0_tm_0, vl);
                    vfloat32m1_t _r01 = vle32_v_f32m1(output0_tm_1, vl);
                    vfloat32m1_t _r02 = vle32_v_f32m1(output0_tm_2, vl);
                    vfloat32m1_t _r03 = vle32_v_f32m1(output0_tm_3, vl);
                    vfloat32m1_t _r04 = vle32_v_f32m1(output0_tm_4, vl);
                    vfloat32m1_t _r05 = vle32_v_f32m1(output0_tm_5, vl);

                    vfloat32m1_t _tmp02a = vfadd_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _tmp13a = vfsub_vv_f32m1(_r01, _r02, vl);

                    vfloat32m1_t _tmp02b = vfadd_vv_f32m1(_r03, _r04, vl);
                    vfloat32m1_t _tmp13b = vfsub_vv_f32m1(_r03, _r04, vl);

                    vfloat32m1_t _tmp0m =
                        vfadd_vv_f32m1(vfadd_vv_f32m1(_r00, _tmp02a, vl), _tmp02b, vl);
                    vfloat32m1_t _tmp1m = vfmacc_vf_f32m1(_tmp13a, 2.f, _tmp13b, vl);
                    vfloat32m1_t _tmp2m = vfmacc_vf_f32m1(_tmp02a, 4.f, _tmp02b, vl);
                    vfloat32m1_t _tmp3m =
                        vfmacc_vf_f32m1(vfadd_vv_f32m1(_r05, _tmp13a, vl), 8.f, _tmp13b, vl);

                    vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                    vse32_v_f32m1(tmp[3][m], _tmp3m, vl);

                    output0_tm_0 += tiles * packn * 6;
                    output0_tm_1 += tiles * packn * 6;
                    output0_tm_2 += tiles * packn * 6;
                    output0_tm_3 += tiles * packn * 6;
                    output0_tm_4 += tiles * packn * 6;
                    output0_tm_5 += tiles * packn * 6;
                }

                for (int m = 0; m < 4; m++) {
                    vfloat32m1_t _tmp00 = vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _tmp01 = vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _tmp02 = vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _tmp03 = vle32_v_f32m1(tmp[m][3], vl);
                    vfloat32m1_t _tmp04 = vle32_v_f32m1(tmp[m][4], vl);
                    vfloat32m1_t _tmp05 = vle32_v_f32m1(tmp[m][5], vl);

                    vfloat32m1_t _tmp02a = vfadd_vv_f32m1(_tmp01, _tmp02, vl);
                    vfloat32m1_t _tmp13a = vfsub_vv_f32m1(_tmp01, _tmp02, vl);

                    vfloat32m1_t _tmp02b = vfadd_vv_f32m1(_tmp03, _tmp04, vl);
                    vfloat32m1_t _tmp13b = vfsub_vv_f32m1(_tmp03, _tmp04, vl);

                    vfloat32m1_t _out00 =
                        vfadd_vv_f32m1(vfadd_vv_f32m1(_tmp00, _tmp02a, vl), _tmp02b, vl);
                    vfloat32m1_t _out01 = vfmacc_vf_f32m1(_tmp13a, 2.f, _tmp13b, vl);
                    vfloat32m1_t _out02 = vfmacc_vf_f32m1(_tmp02a, 4.f, _tmp02b, vl);
                    vfloat32m1_t _out03 =
                        vfmacc_vf_f32m1(vfadd_vv_f32m1(_tmp05, _tmp13a, vl), 8.f, _tmp13b, vl);

                    _out00 = vfadd_vv_f32m1(_bias, _out00, vl);
                    _out01 = vfadd_vv_f32m1(_bias, _out01, vl);
                    _out02 = vfadd_vv_f32m1(_bias, _out02, vl);
                    _out03 = vfadd_vv_f32m1(_bias, _out03, vl);

                    vse32_v_f32m1(output0, _out00, vl);
                    vse32_v_f32m1(output0 + packn * 1, _out01, vl);
                    vse32_v_f32m1(output0 + packn * 2, _out02, vl);
                    vse32_v_f32m1(output0 + packn * 3, _out03, vl);

                    output0 += blk_w * 4 * packn;
                }
            }
        }
    }
}

static inline void wg_bxf3s1_reorder_input_tile8_fp32(const float *src, float *dst, int ch,
                                                      int tiles, int area)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);
    for (int r = 0; r < area; r++) {
        float *img_tm2 = dst + r * tiles * ch;  // input_tm2 r channel data

        int t = 0;
        for (; t + 7 < tiles; t += 8) {
            const float *tm1 = src;
            tm1 += (r * tiles + t) * packn;
            for (int q = 0; q < ch / packn; q++) {
                vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
                vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);
                vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + packn * 2, vl);
                vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + packn * 3, vl);
                vfloat32m1_t _tmp4 = vle32_v_f32m1(tm1 + packn * 4, vl);
                vfloat32m1_t _tmp5 = vle32_v_f32m1(tm1 + packn * 5, vl);
                vfloat32m1_t _tmp6 = vle32_v_f32m1(tm1 + packn * 6, vl);
                vfloat32m1_t _tmp7 = vle32_v_f32m1(tm1 + packn * 7, vl);

                vsseg8e32_v_f32m1(img_tm2, _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7,
                                  vl);
                tm1 += area * tiles * packn;
                img_tm2 += 8 * packn;
            }
        }
        for (; t + 3 < tiles; t += 4) {
            const float *tm1 = src;
            tm1 += (r * tiles + t) * packn;
            for (int q = 0; q < ch / packn; q++) {
                vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
                vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);
                vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + packn * 2, vl);
                vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + packn * 3, vl);

                vsseg4e32_v_f32m1(img_tm2, _tmp0, _tmp1, _tmp2, _tmp3, vl);
                tm1 += area * tiles * packn;
                img_tm2 += 4 * packn;
            }
        }
        for (; t + 1 < tiles; t += 2) {
            const float *tm1 = src;
            tm1 += (r * tiles + t) * packn;
            for (int q = 0; q < ch / packn; q++) {
                vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
                vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);

                vsseg2e32_v_f32m1(img_tm2, _tmp0, _tmp1, vl);
                tm1 += area * tiles * packn;
                img_tm2 += 2 * packn;
            }
        }
        for (; t < tiles; t++) {
            const float *tm1 = src;
            tm1 += (r * tiles + t) * packn;
            for (int q = 0; q < ch / packn; q++) {
                vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);

                vse32_v_f32m1(img_tm2, _tmp0, vl);
                tm1 += area * tiles * packn;
                img_tm2 += 1 * packn;
            }
        }
    }
}

static inline void wg_bxf3s1_batch_gemm_m8n8_fp32(const float *input, const float *kernel,
                                                  float *output, int in_ch, int out_ch, int tiles,
                                                  int area)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int pack2n = packn * 2;
    const int vl = vsetvl_e32m1(packn);

    int p = 0;
    for (; p + pack2n - 1 < out_ch; p += pack2n) {
        float *output0_tm = output + p * area * tiles;  // 8 channel dot output
        float *output1_tm = output0_tm + packn * area * tiles;

        const float *kernel0_tm = kernel + p * area * in_ch;  // 8 channel kernel

        for (int r = 0; r < area; r++) {
            const float *img0 = input + r * tiles * in_ch;  // img_tm2 第r个channel
            int t = 0;
            for (; t + 7 < tiles; t += 8) {
                const float *k0 = kernel0_tm + r * in_ch * pack2n;

                vfloat32m1_t _acc00 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc01 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc02 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc03 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc04 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc05 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc06 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc07 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc10 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc11 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc12 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc13 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc14 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc15 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc16 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc17 = vfmv_v_f_f32m1(0.0f, vl);

                for (int c = 0; c < in_ch; c++) {
                    vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                    vfloat32m1_t _kernel1 = vle32_v_f32m1(k0 + packn, vl);
                    k0 += pack2n;
                    _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                    _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                    _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                    _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                    _acc04 = vfmacc_vf_f32m1(_acc04, img0[4], _kernel0, vl);
                    _acc05 = vfmacc_vf_f32m1(_acc05, img0[5], _kernel0, vl);
                    _acc06 = vfmacc_vf_f32m1(_acc06, img0[6], _kernel0, vl);
                    _acc07 = vfmacc_vf_f32m1(_acc07, img0[7], _kernel0, vl);

                    _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                    _acc11 = vfmacc_vf_f32m1(_acc11, img0[1], _kernel1, vl);
                    _acc12 = vfmacc_vf_f32m1(_acc12, img0[2], _kernel1, vl);
                    _acc13 = vfmacc_vf_f32m1(_acc13, img0[3], _kernel1, vl);
                    _acc14 = vfmacc_vf_f32m1(_acc14, img0[4], _kernel1, vl);
                    _acc15 = vfmacc_vf_f32m1(_acc15, img0[5], _kernel1, vl);
                    _acc16 = vfmacc_vf_f32m1(_acc16, img0[6], _kernel1, vl);
                    _acc17 = vfmacc_vf_f32m1(_acc17, img0[7], _kernel1, vl);
                    img0 += 8;
                }
                vse32_v_f32m1(output0_tm, _acc00, vl);
                vse32_v_f32m1(output0_tm + packn * 1, _acc01, vl);
                vse32_v_f32m1(output0_tm + packn * 2, _acc02, vl);
                vse32_v_f32m1(output0_tm + packn * 3, _acc03, vl);
                vse32_v_f32m1(output0_tm + packn * 4, _acc04, vl);
                vse32_v_f32m1(output0_tm + packn * 5, _acc05, vl);
                vse32_v_f32m1(output0_tm + packn * 6, _acc06, vl);
                vse32_v_f32m1(output0_tm + packn * 7, _acc07, vl);
                output0_tm += packn * 8;

                vse32_v_f32m1(output1_tm, _acc10, vl);
                vse32_v_f32m1(output1_tm + packn * 1, _acc11, vl);
                vse32_v_f32m1(output1_tm + packn * 2, _acc12, vl);
                vse32_v_f32m1(output1_tm + packn * 3, _acc13, vl);
                vse32_v_f32m1(output1_tm + packn * 4, _acc14, vl);
                vse32_v_f32m1(output1_tm + packn * 5, _acc15, vl);
                vse32_v_f32m1(output1_tm + packn * 6, _acc16, vl);
                vse32_v_f32m1(output1_tm + packn * 7, _acc17, vl);
                output1_tm += packn * 8;
            }
            for (; t + 3 < tiles; t += 4) {
                const float *k0 = kernel0_tm + r * in_ch * pack2n;

                vfloat32m1_t _acc00 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc01 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc02 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc03 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc10 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc11 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc12 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc13 = vfmv_v_f_f32m1(0.0f, vl);

                for (int c = 0; c < in_ch; c++) {
                    vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                    vfloat32m1_t _kernel1 = vle32_v_f32m1(k0 + packn, vl);
                    k0 += pack2n;
                    _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                    _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                    _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                    _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);

                    _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                    _acc11 = vfmacc_vf_f32m1(_acc11, img0[1], _kernel1, vl);
                    _acc12 = vfmacc_vf_f32m1(_acc12, img0[2], _kernel1, vl);
                    _acc13 = vfmacc_vf_f32m1(_acc13, img0[3], _kernel1, vl);
                    img0 += 4;
                }
                vse32_v_f32m1(output0_tm, _acc00, vl);
                vse32_v_f32m1(output0_tm + packn * 1, _acc01, vl);
                vse32_v_f32m1(output0_tm + packn * 2, _acc02, vl);
                vse32_v_f32m1(output0_tm + packn * 3, _acc03, vl);
                output0_tm += packn * 4;

                vse32_v_f32m1(output1_tm, _acc10, vl);
                vse32_v_f32m1(output1_tm + packn * 1, _acc11, vl);
                vse32_v_f32m1(output1_tm + packn * 2, _acc12, vl);
                vse32_v_f32m1(output1_tm + packn * 3, _acc13, vl);
                output1_tm += packn * 4;
            }
            for (; t + 1 < tiles; t += 2) {
                const float *k0 = kernel0_tm + r * in_ch * pack2n;

                vfloat32m1_t _acc00 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc01 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc10 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc11 = vfmv_v_f_f32m1(0.0f, vl);

                for (int c = 0; c < in_ch; c++) {
                    vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                    vfloat32m1_t _kernel1 = vle32_v_f32m1(k0 + packn, vl);
                    k0 += pack2n;
                    _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                    _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);

                    _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                    _acc11 = vfmacc_vf_f32m1(_acc11, img0[1], _kernel1, vl);
                    img0 += 2;
                }
                vse32_v_f32m1(output0_tm, _acc00, vl);
                vse32_v_f32m1(output0_tm + packn * 1, _acc01, vl);
                output0_tm += packn * 2;

                vse32_v_f32m1(output1_tm, _acc10, vl);
                vse32_v_f32m1(output1_tm + packn * 1, _acc11, vl);
                output1_tm += packn * 2;
            }
            for (; t < tiles; t++) {
                const float *k0 = kernel0_tm + r * in_ch * pack2n;

                vfloat32m1_t _acc00 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc10 = vfmv_v_f_f32m1(0.0f, vl);

                for (int c = 0; c < in_ch; c++) {
                    vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                    vfloat32m1_t _kernel1 = vle32_v_f32m1(k0 + packn, vl);
                    k0 += pack2n;
                    _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                    _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                    img0 += 1;
                }
                vse32_v_f32m1(output0_tm, _acc00, vl);
                output0_tm += packn * 1;

                vse32_v_f32m1(output1_tm, _acc10, vl);
                output1_tm += packn * 1;
            }
        }
    }

    for (; p + packn - 1 < out_ch; p += packn) {
        float *output0_tm = output + p * area * tiles;        // 4 channel dot output
        const float *kernel0_tm = kernel + p * area * in_ch;  // 4 channel kernel

        for (int r = 0; r < area; r++) {
            const float *img0 = input + r * tiles * in_ch;  // img_tm2 第r个channel
            int t = 0;
            for (; t + 7 < tiles; t += 8) {
                const float *k0 = kernel0_tm + r * in_ch * packn;

                vfloat32m1_t _acc00 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc01 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc02 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc03 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc04 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc05 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc06 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc07 = vfmv_v_f_f32m1(0.0f, vl);

                for (int c = 0; c < in_ch; c++) {
                    vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                    k0 += packn;
                    _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                    _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                    _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                    _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                    _acc04 = vfmacc_vf_f32m1(_acc04, img0[4], _kernel0, vl);
                    _acc05 = vfmacc_vf_f32m1(_acc05, img0[5], _kernel0, vl);
                    _acc06 = vfmacc_vf_f32m1(_acc06, img0[6], _kernel0, vl);
                    _acc07 = vfmacc_vf_f32m1(_acc07, img0[7], _kernel0, vl);
                    img0 += 8;
                }
                vse32_v_f32m1(output0_tm, _acc00, vl);
                vse32_v_f32m1(output0_tm + packn * 1, _acc01, vl);
                vse32_v_f32m1(output0_tm + packn * 2, _acc02, vl);
                vse32_v_f32m1(output0_tm + packn * 3, _acc03, vl);
                vse32_v_f32m1(output0_tm + packn * 4, _acc04, vl);
                vse32_v_f32m1(output0_tm + packn * 5, _acc05, vl);
                vse32_v_f32m1(output0_tm + packn * 6, _acc06, vl);
                vse32_v_f32m1(output0_tm + packn * 7, _acc07, vl);
                output0_tm += packn * 8;
            }
            for (; t + 3 < tiles; t += 4) {
                const float *k0 = kernel0_tm + r * in_ch * packn;

                vfloat32m1_t _acc00 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc01 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc02 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc03 = vfmv_v_f_f32m1(0.0f, vl);

                for (int c = 0; c < in_ch; c++) {
                    vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                    k0 += packn;
                    _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                    _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                    _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                    _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                    img0 += 4;
                }
                vse32_v_f32m1(output0_tm, _acc00, vl);
                vse32_v_f32m1(output0_tm + packn * 1, _acc01, vl);
                vse32_v_f32m1(output0_tm + packn * 2, _acc02, vl);
                vse32_v_f32m1(output0_tm + packn * 3, _acc03, vl);
                output0_tm += packn * 4;
            }
            for (; t + 1 < tiles; t += 2) {
                const float *k0 = kernel0_tm + r * in_ch * packn;

                vfloat32m1_t _acc00 = vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t _acc01 = vfmv_v_f_f32m1(0.0f, vl);

                for (int c = 0; c < in_ch; c++) {
                    vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                    k0 += packn;
                    _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                    _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                    img0 += 2;
                }
                vse32_v_f32m1(output0_tm, _acc00, vl);
                vse32_v_f32m1(output0_tm + packn * 1, _acc01, vl);
                output0_tm += packn * 2;
            }
            for (; t < tiles; t++) {
                const float *k0 = kernel0_tm + r * in_ch * packn;

                vfloat32m1_t _acc00 = vfmv_v_f_f32m1(0.0f, vl);

                for (int c = 0; c < in_ch; c++) {
                    vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                    k0 += packn;
                    _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                    img0 += 1;
                }
                vse32_v_f32m1(output0_tm, _acc00, vl);
                output0_tm += packn * 1;
            }
        }
    }
}

static inline void wg_b6f3s1_trans_input_packn_fp32(const float *src, float *dst, int ch, int h,
                                                    int w, int blk_h, int blk_w)
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
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);
    int tiles = blk_h * blk_w;
    for (int q = 0; q + packn - 1 < ch; q += packn) {
        const float *img0 = src + q * h * w;    // feature map after padding - q channel
        float *img0_tm = dst + q * 64 * tiles;  // transform and interleave - q channel

        float tmp[8][8][packn];

        for (int i = 0; i < blk_h; i++) {
            for (int j = 0; j < blk_w; j++) {
                const float *r0 =
                    img0 + (i * w * 6 + j * 6) * packn;  // feature map after padding 8*8 start addr
                float *r0_tm = img0_tm + (i * blk_w + j) * packn;  // input_tm1 8*8 block start addr

                for (int m = 0; m < 8; m++) {
                    vfloat32m1_t _r00 = vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r01 = vle32_v_f32m1(r0 + packn * 1, vl);
                    vfloat32m1_t _r02 = vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _r03 = vle32_v_f32m1(r0 + packn * 3, vl);
                    vfloat32m1_t _r04 = vle32_v_f32m1(r0 + packn * 4, vl);
                    vfloat32m1_t _r05 = vle32_v_f32m1(r0 + packn * 5, vl);
                    vfloat32m1_t _r06 = vle32_v_f32m1(r0 + packn * 6, vl);
                    vfloat32m1_t _r07 = vle32_v_f32m1(r0 + packn * 7, vl);

                    vfloat32m1_t _tmp0m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r00, _r06, vl), 5.25f,
                                                          vfsub_vv_f32m1(_r04, _r02, vl), vl);
                    vfloat32m1_t _tmp7m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r07, _r01, vl), 5.25f,
                                                          vfsub_vv_f32m1(_r03, _r05, vl), vl);

                    vfloat32m1_t _tmp12a =
                        vfmacc_vf_f32m1(vfadd_vv_f32m1(_r02, _r06, vl), -4.25f, _r04, vl);
                    vfloat32m1_t _tmp12b =
                        vfmacc_vf_f32m1(vfadd_vv_f32m1(_r01, _r05, vl), -4.25f, _r03, vl);
                    vfloat32m1_t _tmp1m = vfadd_vv_f32m1(_tmp12a, _tmp12b, vl);
                    vfloat32m1_t _tmp2m = vfsub_vv_f32m1(_tmp12a, _tmp12b, vl);

                    vfloat32m1_t _tmp34a =
                        vfmacc_vf_f32m1(vfmacc_vf_f32m1(_r06, 0.25f, _r02, vl), -1.25f, _r04, vl);
                    vfloat32m1_t _tmp34b = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(vfmul_vf_f32m1(_r01, 0.5f, vl), -2.5f, _r03, vl), 2.f, _r05,
                        vl);
                    vfloat32m1_t _tmp3m = vfadd_vv_f32m1(_tmp34a, _tmp34b, vl);
                    vfloat32m1_t _tmp4m = vfsub_vv_f32m1(_tmp34a, _tmp34b, vl);

                    vfloat32m1_t _tmp56a =
                        vfmacc_vf_f32m1(_r06, 4.f, vfmacc_vf_f32m1(_r02, -1.25f, _r04, vl), vl);
                    vfloat32m1_t _tmp56b = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(vfmul_vf_f32m1(_r01, 2.f, vl), -2.5f, _r03, vl), 0.5f, _r05,
                        vl);
                    vfloat32m1_t _tmp5m = vfadd_vv_f32m1(_tmp56a, _tmp56b, vl);
                    vfloat32m1_t _tmp6m = vfsub_vv_f32m1(_tmp56a, _tmp56b, vl);

                    vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    vse32_v_f32m1(tmp[7][m], _tmp7m, vl);
                    vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                    vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                    vse32_v_f32m1(tmp[4][m], _tmp4m, vl);
                    vse32_v_f32m1(tmp[5][m], _tmp5m, vl);
                    vse32_v_f32m1(tmp[6][m], _tmp6m, vl);

                    r0 += w * packn;
                }

                for (int m = 0; m < 8; m++) {
                    float *r0_tm0 = r0_tm;
                    float *r0_tm1 = r0_tm0 + tiles * packn;
                    float *r0_tm2 = r0_tm1 + tiles * packn;
                    float *r0_tm3 = r0_tm2 + tiles * packn;
                    float *r0_tm4 = r0_tm3 + tiles * packn;
                    float *r0_tm5 = r0_tm4 + tiles * packn;
                    float *r0_tm6 = r0_tm5 + tiles * packn;
                    float *r0_tm7 = r0_tm6 + tiles * packn;

                    vfloat32m1_t _tmp00 = vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _tmp01 = vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _tmp02 = vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _tmp03 = vle32_v_f32m1(tmp[m][3], vl);
                    vfloat32m1_t _tmp04 = vle32_v_f32m1(tmp[m][4], vl);
                    vfloat32m1_t _tmp05 = vle32_v_f32m1(tmp[m][5], vl);
                    vfloat32m1_t _tmp06 = vle32_v_f32m1(tmp[m][6], vl);
                    vfloat32m1_t _tmp07 = vle32_v_f32m1(tmp[m][7], vl);

                    vfloat32m1_t _r0tm0 = vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp00, _tmp06, vl), 5.25f,
                                                          vfsub_vv_f32m1(_tmp04, _tmp02, vl), vl);
                    vfloat32m1_t _r0tm7 = vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp07, _tmp01, vl), 5.25f,
                                                          vfsub_vv_f32m1(_tmp03, _tmp05, vl), vl);

                    vfloat32m1_t _tmp12a =
                        vfmacc_vf_f32m1(vfadd_vv_f32m1(_tmp02, _tmp06, vl), -4.25f, _tmp04, vl);
                    vfloat32m1_t _tmp12b =
                        vfmacc_vf_f32m1(vfadd_vv_f32m1(_tmp01, _tmp05, vl), -4.25f, _tmp03, vl);
                    vfloat32m1_t _r0tm1 = vfadd_vv_f32m1(_tmp12a, _tmp12b, vl);
                    vfloat32m1_t _r0tm2 = vfsub_vv_f32m1(_tmp12a, _tmp12b, vl);

                    vfloat32m1_t _tmp34a = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(_tmp06, 0.25f, _tmp02, vl), -1.25f, _tmp04, vl);
                    vfloat32m1_t _tmp34b = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(vfmul_vf_f32m1(_tmp01, 0.5f, vl), -2.5f, _tmp03, vl), 2.f,
                        _tmp05, vl);
                    vfloat32m1_t _r0tm3 = vfadd_vv_f32m1(_tmp34a, _tmp34b, vl);
                    vfloat32m1_t _r0tm4 = vfsub_vv_f32m1(_tmp34a, _tmp34b, vl);

                    vfloat32m1_t _tmp56a = vfmacc_vf_f32m1(
                        _tmp06, 4.f, vfmacc_vf_f32m1(_tmp02, -1.25f, _tmp04, vl), vl);
                    vfloat32m1_t _tmp56b = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(vfmul_vf_f32m1(_tmp01, 2.f, vl), -2.5f, _tmp03, vl), 0.5f,
                        _tmp05, vl);
                    vfloat32m1_t _r0tm5 = vfadd_vv_f32m1(_tmp56a, _tmp56b, vl);
                    vfloat32m1_t _r0tm6 = vfsub_vv_f32m1(_tmp56a, _tmp56b, vl);

                    vse32_v_f32m1(r0_tm0, _r0tm0, vl);
                    vse32_v_f32m1(r0_tm7, _r0tm7, vl);
                    vse32_v_f32m1(r0_tm1, _r0tm1, vl);
                    vse32_v_f32m1(r0_tm2, _r0tm2, vl);
                    vse32_v_f32m1(r0_tm3, _r0tm3, vl);
                    vse32_v_f32m1(r0_tm4, _r0tm4, vl);
                    vse32_v_f32m1(r0_tm5, _r0tm5, vl);
                    vse32_v_f32m1(r0_tm6, _r0tm6, vl);

                    r0_tm += tiles * packn * 8;
                }
            }
        }
    }
}

static inline void wg_b6f3s1_trans_output_packn_fp32(const float *src, const float *bias,
                                                     float *dst, int ch, int blk_h, int blk_w)
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
    AT = {
        { 1  1  1   1    1   32    32   0 };
        { 0  1  -1  2   -2   16   -16   0 };
        { 0  1  1   4    4   8     8    0 };
        { 0  1  -1  8   -8   4    -4    0 };
        { 0  1  1   16  16   2     2    0 };
        { 0  1  -1  32  -32  1    -1    1 }
    };
    */
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);
    int tiles = blk_h * blk_w;
    for (int p = 0; p + packn - 1 < ch; p += packn) {
        const float *out0_tm = src + p * 64 * tiles;    // 输出转换前/dot后 第p个channel
        float *out0 = dst + p * 6 * blk_h * 6 * blk_w;  // 转换后输出 第p个channel

        float tmp[6][8][packn];

        vfloat32m1_t _bias = bias ? vle32_v_f32m1(bias + p, vl) : vfmv_v_f_f32m1(0.0f, vl);

        for (int i = 0; i < blk_h; i++) {
            for (int j = 0; j < blk_w; j++) {
                const float *output0_tm_0 = out0_tm + (i * blk_w + j) * packn;  // 8*8 起始地址
                const float *output0_tm_1 = output0_tm_0 + tiles * packn * 1;
                const float *output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const float *output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                const float *output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                const float *output0_tm_5 = output0_tm_0 + tiles * packn * 5;
                const float *output0_tm_6 = output0_tm_0 + tiles * packn * 6;
                const float *output0_tm_7 = output0_tm_0 + tiles * packn * 7;

                float *output0 = out0 + (i * blk_w * 6 * 6 + j * 6) * packn;  // out 6*6 addr

                for (int m = 0; m < 8; m++) {
                    vfloat32m1_t _r00 = vle32_v_f32m1(output0_tm_0, vl);
                    vfloat32m1_t _r01 = vle32_v_f32m1(output0_tm_1, vl);
                    vfloat32m1_t _r02 = vle32_v_f32m1(output0_tm_2, vl);
                    vfloat32m1_t _r03 = vle32_v_f32m1(output0_tm_3, vl);
                    vfloat32m1_t _r04 = vle32_v_f32m1(output0_tm_4, vl);
                    vfloat32m1_t _r05 = vle32_v_f32m1(output0_tm_5, vl);
                    vfloat32m1_t _r06 = vle32_v_f32m1(output0_tm_6, vl);
                    vfloat32m1_t _r07 = vle32_v_f32m1(output0_tm_7, vl);

                    vfloat32m1_t _tmp024a = vfadd_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _tmp135a = vfsub_vv_f32m1(_r01, _r02, vl);

                    vfloat32m1_t _tmp024b = vfadd_vv_f32m1(_r03, _r04, vl);
                    vfloat32m1_t _tmp135b = vfsub_vv_f32m1(_r03, _r04, vl);

                    vfloat32m1_t _tmp024c = vfadd_vv_f32m1(_r05, _r06, vl);
                    vfloat32m1_t _tmp135c = vfsub_vv_f32m1(_r05, _r06, vl);

                    vfloat32m1_t _tmp0m =
                        vfadd_vv_f32m1(vfadd_vv_f32m1(_r00, _tmp024a, vl),
                                       vfmacc_vf_f32m1(_tmp024b, 32.f, _tmp024c, vl), vl);
                    vfloat32m1_t _tmp2m = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(_tmp024a, 4.f, _tmp024b, vl), 8.f, _tmp024c, vl);
                    vfloat32m1_t _tmp4m = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(_tmp024a, 16.f, _tmp024b, vl), 2.f, _tmp024c, vl);

                    vfloat32m1_t _tmp1m = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(_tmp135a, 2.f, _tmp135b, vl), 16.f, _tmp135c, vl);
                    vfloat32m1_t _tmp3m = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(_tmp135a, 8.f, _tmp135b, vl), 4.f, _tmp135c, vl);
                    vfloat32m1_t _tmp5m =
                        vfadd_vv_f32m1(vfadd_vv_f32m1(_r07, _tmp135a, vl),
                                       vfmacc_vf_f32m1(_tmp135c, 32.f, _tmp135b, vl), vl);

                    vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                    vse32_v_f32m1(tmp[4][m], _tmp4m, vl);
                    vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                    vse32_v_f32m1(tmp[5][m], _tmp5m, vl);

                    output0_tm_0 += tiles * packn * 8;
                    output0_tm_1 += tiles * packn * 8;
                    output0_tm_2 += tiles * packn * 8;
                    output0_tm_3 += tiles * packn * 8;
                    output0_tm_4 += tiles * packn * 8;
                    output0_tm_5 += tiles * packn * 8;
                    output0_tm_6 += tiles * packn * 8;
                    output0_tm_7 += tiles * packn * 8;
                }

                for (int m = 0; m < 6; m++) {
                    vfloat32m1_t _tmp00 = vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _tmp01 = vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _tmp02 = vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _tmp03 = vle32_v_f32m1(tmp[m][3], vl);
                    vfloat32m1_t _tmp04 = vle32_v_f32m1(tmp[m][4], vl);
                    vfloat32m1_t _tmp05 = vle32_v_f32m1(tmp[m][5], vl);
                    vfloat32m1_t _tmp06 = vle32_v_f32m1(tmp[m][6], vl);
                    vfloat32m1_t _tmp07 = vle32_v_f32m1(tmp[m][7], vl);

                    vfloat32m1_t _tmp024a = vfadd_vv_f32m1(_tmp01, _tmp02, vl);
                    vfloat32m1_t _tmp135a = vfsub_vv_f32m1(_tmp01, _tmp02, vl);

                    vfloat32m1_t _tmp024b = vfadd_vv_f32m1(_tmp03, _tmp04, vl);
                    vfloat32m1_t _tmp135b = vfsub_vv_f32m1(_tmp03, _tmp04, vl);

                    vfloat32m1_t _tmp024c = vfadd_vv_f32m1(_tmp05, _tmp06, vl);
                    vfloat32m1_t _tmp135c = vfsub_vv_f32m1(_tmp05, _tmp06, vl);

                    vfloat32m1_t _output00 =
                        vfadd_vv_f32m1(vfadd_vv_f32m1(_tmp00, _tmp024a, vl),
                                       vfmacc_vf_f32m1(_tmp024b, 32.f, _tmp024c, vl), vl);
                    vfloat32m1_t _output02 = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(_tmp024a, 4.f, _tmp024b, vl), 8.f, _tmp024c, vl);
                    vfloat32m1_t _output04 = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(_tmp024a, 16.f, _tmp024b, vl), 2.f, _tmp024c, vl);

                    vfloat32m1_t _output01 = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(_tmp135a, 2.f, _tmp135b, vl), 16.f, _tmp135c, vl);
                    vfloat32m1_t _output03 = vfmacc_vf_f32m1(
                        vfmacc_vf_f32m1(_tmp135a, 8.f, _tmp135b, vl), 4.f, _tmp135c, vl);
                    vfloat32m1_t _output05 =
                        vfadd_vv_f32m1(vfadd_vv_f32m1(_tmp07, _tmp135a, vl),
                                       vfmacc_vf_f32m1(_tmp135c, 32.f, _tmp135b, vl), vl);

                    _output00 = vfadd_vv_f32m1(_bias, _output00, vl);
                    _output01 = vfadd_vv_f32m1(_bias, _output01, vl);
                    _output02 = vfadd_vv_f32m1(_bias, _output02, vl);
                    _output03 = vfadd_vv_f32m1(_bias, _output03, vl);
                    _output04 = vfadd_vv_f32m1(_bias, _output04, vl);
                    _output05 = vfadd_vv_f32m1(_bias, _output05, vl);

                    vse32_v_f32m1(output0, _output00, vl);
                    vse32_v_f32m1(output0 + packn * 2, _output02, vl);
                    vse32_v_f32m1(output0 + packn * 4, _output04, vl);
                    vse32_v_f32m1(output0 + packn * 1, _output01, vl);
                    vse32_v_f32m1(output0 + packn * 3, _output03, vl);
                    vse32_v_f32m1(output0 + packn * 5, _output05, vl);

                    output0 += blk_w * 6 * packn;
                }
            }
        }
    }
}

/******************************************************************************************
 * kernel layout before:  [O, I, 3, 3]
 * kernel layout after :  [O/pack2n, 36, I, pack2n] --> [O/packn, 36, I, packn]
 * constrain: output channel % packn = 0
 *            input channel % packn = 0
 ******************************************************************************************/
void shl_rvv_wg_b4f3s1_trans_kernel_packn_fp32(struct csinn_tensor *src_kernel,
                                               struct csinn_tensor *dst_kernel)
{
    int32_t outch = src_kernel->dim[0];
    int32_t inch = src_kernel->dim[1];

    float *kernel_data = (float *)src_kernel->data;
    // for kernel transform buf, 3x3 --> 6x6
    float *kernel_tm = (float *)shl_mem_alloc(outch * inch * 6 * 6 * sizeof(float));

    // kernel transform matrix: G
    const float ktm[6][3] = {{1.0f / 4, 0.0f, 0.0f},
                             {-1.0f / 6, -1.0f / 6, -1.0f / 6},
                             {-1.0f / 6, 1.0f / 6, -1.0f / 6},
                             {1.0f / 24, 1.0f / 12, 1.0f / 6},
                             {1.0f / 24, -1.0f / 12, 1.0f / 6},
                             {0.0f, 0.0f, 1.0f}};

    csinn_tensor_copy(dst_kernel, src_kernel);

    for (int p = 0; p < outch; p++) {
        for (int q = 0; q < inch; q++) {
            const float *kernel0 = kernel_data + p * inch * 9 + q * 9;
            float *kernel_tm0 = kernel_tm + p * inch * 36 + q * 36;

            // transform kernel
            const float *k0 = kernel0;
            const float *k1 = kernel0 + 3;
            const float *k2 = kernel0 + 6;

            // h : first compute the transport matrix tmp = (g * GT)T
            float tmp[6][3];
            for (int i = 0; i < 6; i++) {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 6; j++) {
                float *tmpp = &tmp[j][0];

                for (int i = 0; i < 6; i++) {
                    kernel_tm0[j * 6 + i] =
                        tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // optimized layout for winograd b4f3
    // [O, I, 6, 6]  -->  [O/pack2n, 6*6, I, pack2n]
    float *kernel_tm_packn = (float *)shl_mem_alloc(outch / 4 * 36 * inch * 4 * sizeof(float));
    dst_kernel->data = kernel_tm_packn;

    const int packn = csrr_vlenb() / sizeof(float);
    const int pack2n = packn * 2;

    int oc = 0;
    for (; oc + pack2n - 1 < outch; oc += pack2n) {
        float *g0 = kernel_tm_packn + oc * 36 * inch;
        for (int k = 0; k < 36; k++) {
            float *g00 = g0 + k * inch * pack2n;
            for (int ic = 0; ic < inch; ic++) {
                for (int j = 0; j < pack2n; j++) {
                    float *k00 = kernel_tm + (oc + j) * 36 * inch + ic * 36;
                    *g00++ = k00[k];
                }
            }
        }
    }
    // [O/packn, 6*6, I, packn]
    for (; oc + packn - 1 < outch; oc += packn) {
        float *g0 = kernel_tm_packn + oc * 36 * inch;
        for (int k = 0; k < 36; k++) {
            float *g00 = g0 + k * inch * packn;
            for (int ic = 0; ic < inch; ic++) {
                for (int j = 0; j < packn; j++) {
                    float *k00 = kernel_tm + (oc + j) * 36 * inch + ic * 36;
                    *g00++ = k00[k];
                }
            }
        }
    }
    shl_mem_free(kernel_tm);
}

/******************************************************************************************
 * constrain: output channel % packn = 0
 *            input channel % packn = 0
 ******************************************************************************************/
int shl_rvv_wg_b4f3s1_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)params->conv_extra.kernel_tm->data;
    float *bias_data = (float *)bias->data;

    // param
    int pad_left = params->pad_left;
    int pad_top = params->pad_top;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int input_size = in_c * in_h * in_w;

    int out_c = kernel->dim[0];
    int out_h = output->dim[2];
    int out_w = output->dim[3];
    int output_size = out_c * out_h * out_w;

    // winograd param
    int block_h = (out_h + 3) / 4;
    int block_w = (out_w + 3) / 4;

    // block * 4 for alignment with 4，kernel = 3 * 3 ，stride = 1，thus input_size + 2
    int padded_in_h = block_h * 4 + 2;
    int padded_in_w = block_w * 4 + 2;
    int padded_in_hw = padded_in_h * padded_in_w;  // element size after padding per channel

    int tiles = block_h * block_w;

    for (int n = 0; n < batch; n++) {
        // pad buffer: [in_c/packn h w packn]
        float *input_padd_buf = (float *)shl_mem_alloc(in_c * padded_in_hw * sizeof(float));

        // pad input
        winograd_pad_input_packn_fp32(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h,
                                      padded_in_w, pad_top, pad_left);

        input_data += input_size;

        /****************************** transform input *****************************/
        // input transform buffer1: [in_c/packn, 36, tiles, packn]
        float *input_tm1_buf = (float *)shl_mem_alloc(in_c / 4 * 36 * tiles * 4 * sizeof(float));
        wg_b4f3s1_trans_input_packn_fp32(input_padd_buf, input_tm1_buf, in_c, padded_in_h,
                                         padded_in_w, block_h, block_w);
        shl_mem_free(input_padd_buf);

        /****************************** reorder input_tm1_buf *****************************/
        // input reorder buffer2: [36, tiles/8, in_c, 8]
        float *input_tm2_buf = (float *)shl_mem_alloc(36 * tiles * in_c * sizeof(float));
        wg_bxf3s1_reorder_input_tile8_fp32(input_tm1_buf, input_tm2_buf, in_c, tiles, 36);
        shl_mem_free(input_tm1_buf);

        /****************************** batch gemm *****************************/
        // output_dot_buf： [out_c/packn, 36, tiles, packn]
        float *output_dot_buf = (float *)shl_mem_alloc(out_c / 4 * 36 * tiles * 4 * sizeof(float));
        wg_bxf3s1_batch_gemm_m8n8_fp32(input_tm2_buf, kernel_data, output_dot_buf, in_c, out_c,
                                       tiles, 36);
        shl_mem_free(input_tm2_buf);

        /****************************** transform output *****************************/
        // output_tm1_buf: [out_c/packn, out_h4, out_w4, packn]
        float *output_tm1_buf =
            (float *)shl_mem_alloc(out_c / 4 * tiles * 4 * 4 * 4 * sizeof(float));
        wg_b4f3s1_trans_output_packn_fp32(output_dot_buf, bias_data, output_tm1_buf, out_c, block_h,
                                          block_w);
        shl_mem_free(output_dot_buf);

        // crop the output after transform: cut extra part (right , bottom)
        winograd_crop_output_packn_fp32(output_tm1_buf, output_data, out_c, out_h, out_w,
                                        block_h * 4, block_w * 4);
        output_data += output_size;
        shl_mem_free(output_tm1_buf);
    }
    return CSINN_TRUE;
}

/******************************************************************************************
 * kernel layout before:  [O, I, 3, 3]
 * kernel layout after :  [O/pack2n, 36, I, pack2n] --> [O/packn, 36, I, packn]
 * constrain: output channel % packn = 0
 *            input channel % packn = 0
 ******************************************************************************************/
void shl_rvv_wg_b6f3s1_trans_kernel_packn_fp32(struct csinn_tensor *src_kernel,
                                               struct csinn_tensor *dst_kernel)
{
    int32_t outch = src_kernel->dim[0];
    int32_t inch = src_kernel->dim[1];

    float *kernel_data = (float *)src_kernel->data;
    // for kernel transform buf, 3x3 --> 8x8
    float *kernel_tm = (float *)shl_mem_alloc(outch * inch * 8 * 8 * sizeof(float));
    // kernel transform matrix: G
    const float ktm[8][3] = {{1.0f, 0.0f, 0.0f},
                             {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                             {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                             {1.0f / 90, 1.0f / 45, 2.0f / 45},
                             {1.0f / 90, -1.0f / 45, 2.0f / 45},
                             {1.0f / 45, 1.0f / 90, 1.0f / 180},
                             {1.0f / 45, -1.0f / 90, 1.0f / 180},
                             {0.0f, 0.0f, 1.0f}};

    // const float ktm[8][3] = {
    //     {1.0f, 0.0f, 0.0f},
    //     {-2.0f / 9, -2.0f / 9, -2.0f / 9},
    //     {-2.0f / 9, 2.0f / 9, -2.0f / 9},
    //     {1.0f / 90, 1.0f / 45, 2.0f / 45},
    //     {1.0f / 90, -1.0f / 45, 2.0f / 45},
    //     {32.0f / 45, 16.0f / 45, 8.0f / 45},
    //     {32.0f / 45, -16.0f / 45, 8.0f / 45},
    //     {0.0f, 0.0f, 1.0f}
    // };

    csinn_tensor_copy(dst_kernel, src_kernel);

    for (int p = 0; p < outch; p++) {
        for (int q = 0; q < inch; q++) {
            const float *kernel0 = kernel_data + p * inch * 9 + q * 9;
            float *kernel_tmp = kernel_tm + p * inch * 64 + q * 64;

            // transform kernel
            const float *k0 = kernel0;
            const float *k1 = kernel0 + 3;
            const float *k2 = kernel0 + 6;

            // h : first compute the transport matrix tmp = (g * GT)T
            float tmp[8][3];
            for (int i = 0; i < 8; i++) {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 8; j++) {
                float *tmpp = &tmp[j][0];

                for (int i = 0; i < 8; i++) {
                    kernel_tmp[j * 8 + i] =
                        tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
    // optimized layout for winograd64
    float *kernel_tm_packn = (float *)shl_mem_alloc(64 * outch / 4 * inch * 4 * sizeof(float));
    dst_kernel->data = kernel_tm_packn;

    const int packn = csrr_vlenb() / sizeof(float);
    const int pack2n = packn * 2;

    int oc = 0;
    for (; oc + pack2n - 1 < outch; oc += pack2n) {
        float *g0 = kernel_tm_packn + oc * 64 * inch;
        for (int k = 0; k < 64; k++) {
            float *g00 = g0 + k * inch * pack2n;
            for (int ic = 0; ic < inch; ic++) {
                for (int j = 0; j < pack2n; j++) {
                    float *k00 = kernel_tm + (oc + j) * 64 * inch + ic * 64;
                    *g00++ = k00[k];
                }
            }
        }
    }

    for (; oc + packn - 1 < outch; oc += packn) {
        float *g0 = kernel_tm_packn + oc * 64 * inch;
        for (int k = 0; k < 64; k++) {
            float *g00 = g0 + k * inch * packn;
            for (int ic = 0; ic < inch; ic++) {
                for (int j = 0; j < packn; j++) {
                    float *k00 = kernel_tm + (oc + j) * 64 * inch + ic * 64;
                    *g00++ = k00[k];
                }
            }
        }
    }
    shl_mem_free(kernel_tm);
}

/******************************************************************************************
 * constrain: output channel % packn = 0
 *            input channel % packn = 0
 ******************************************************************************************/
int shl_rvv_wg_b6f3s1_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)params->conv_extra.kernel_tm->data;
    float *bias_data = (float *)bias->data;

    // param
    int pad_left = params->pad_left;
    int pad_top = params->pad_top;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int input_size = in_c * in_h * in_w;

    int out_c = kernel->dim[0];
    int out_h = output->dim[2];
    int out_w = output->dim[3];
    int output_size = out_c * out_h * out_w;

    // winograd param
    int block_h = (out_h + 5) / 6;
    int block_w = (out_w + 5) / 6;

    // block * 6 for alignment with 6, kernel = 3 * 3, stride = 1, thus input_size + 2
    int padded_in_h = block_h * 6 + 2;
    int padded_in_w = block_w * 6 + 2;
    int padded_in_hw = padded_in_h * padded_in_w;  // element size after padding per channel

    int tiles = block_h * block_w;

    for (int n = 0; n < batch; n++) {
        // pad buffer: [in_c/packn h w packn]
        float *input_padd_buf = (float *)shl_mem_alloc(in_c * padded_in_hw * sizeof(float));

        // pad input
        winograd_pad_input_packn_fp32(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h,
                                      padded_in_w, pad_top, pad_left);

        input_data += input_size;

        /****************************** transform input *****************************/
        // input transform buffer1: [in_ch/packn, 64, tiles, packn]
        float *input_tm1_buf = (float *)shl_mem_alloc(in_c / 4 * 64 * tiles * 4 * sizeof(float));
        wg_b6f3s1_trans_input_packn_fp32(input_padd_buf, input_tm1_buf, in_c, padded_in_h,
                                         padded_in_w, block_h, block_w);
        shl_mem_free(input_padd_buf);

        /****************************** reorder input_tm1_buf *****************************/
        // input reorder buffer2: [64, tiles/8, in_c, 8]
        float *input_tm2_buf = (float *)shl_mem_alloc(64 * tiles * in_c * sizeof(float));
        wg_bxf3s1_reorder_input_tile8_fp32(input_tm1_buf, input_tm2_buf, in_c, tiles, 64);
        shl_mem_free(input_tm1_buf);

        /****************************** batch gemm *****************************/
        // output_dot_buf： [out_c/packn, 64, tiles, packn]
        float *output_dot_buf = (float *)shl_mem_alloc(out_c / 4 * 64 * tiles * 4 * sizeof(float));
        wg_bxf3s1_batch_gemm_m8n8_fp32(input_tm2_buf, kernel_data, output_dot_buf, in_c, out_c,
                                       tiles, 64);
        shl_mem_free(input_tm2_buf);

        /****************************** transform output *****************************/
        // output_tm1_buf: [out_c/packn, out_h4, out_w4, packn]
        float *output_tm1_buf =
            (float *)shl_mem_alloc(out_c / 4 * tiles * 6 * 6 * 4 * sizeof(float));
        wg_b6f3s1_trans_output_packn_fp32(output_dot_buf, bias_data, output_tm1_buf, out_c, block_h,
                                          block_w);
        shl_mem_free(output_dot_buf);

        // crop the output after transform: cut extra part (right , bottom)
        winograd_crop_output_packn_fp32(output_tm1_buf, output_data, out_c, out_h, out_w,
                                        block_h * 6, block_w * 6);
        output_data += output_size;
        shl_mem_free(output_tm1_buf);
    }
    return CSINN_TRUE;
}
