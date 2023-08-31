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

#include "c908/c908.h"

/*************************************************************
    note: VLEN = 128
*************************************************************/

/******************************************************************************************
 * padding input for winograd input transform , and change memory layout
 * input layout: [n c h w]
 * input_padded layout: [n, c/8, h, w, 8]
 * constrain: input channel % 8 = 0
 ******************************************************************************************/
static void winograd_pad_input_packn_int8(const int8_t *input, int8_t *input_padded, int inc,
                                          int inh, int inw, int padded_h, int padded_w, int pad_top,
                                          int pad_left, int8_t pad_value)
{
    shl_rvv_pad_input_packn_int8(input, input_padded, inc, inh, inw, padded_h, padded_w, pad_top,
                                 pad_left, pad_value);
}

/******************************************************************************************
 * cut winograd output transform for output, and change memory layout
 * winograd output transform layout: [n, c/8, h, w, 8]
 * output layout: [n, c, h, w]
 * constrain: output channel % 8 = 0
 ******************************************************************************************/
static void winograd_crop_output_packn_int8(const int8_t *output_trans, int8_t *output, int out_c,
                                            int out_h, int out_w, int wino_h, int wino_w)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    const int out_size = out_h * out_w;  // per-channel size
    const int crop_size = wino_h * wino_w;

    int c = 0;
    for (; c + packn - 1 < out_c; c += packn) {
        int8_t *out_tm_ptr = (int8_t *)output_trans + c * crop_size;
        int8_t *out_ptr = output + c * out_size;

        for (int h = 0; h < out_h; h++) {
            int8_t *crop_ptr = out_tm_ptr + h * wino_w * packn;
            for (int w = 0; w < out_w; w++) {
                vint8m1_t _tmp = vle8_v_i8m1(crop_ptr, vl);
                crop_ptr += packn;
                vse8_v_i8m1(out_ptr, _tmp, vl);
                out_ptr += packn;
            }
        }
    }
}

/******************************************************************************************
 * winograd int8 postprocess  int32 --> int8
 * _src: 8 channels int32 macc
 * multiplier: multi for scale, support channel quantization
 * shift: shift for scale, support channel quantization
 * out_zp: output zero_point
 ******************************************************************************************/
#ifdef RVV_1_0_0
static vint8mf2_t requantize_m2_s(vint32m2_t _src, int32_t *multiplier, int32_t *shift,
                                  int32_t out_zp, int vl)
{
    vint32m2_t _mult = vle32_v_i32m2(multiplier, vl);
    vint32m2_t _shift = vle32_v_i32m2(shift, vl);
    vint32m2_t _mulh = vmulh_vv_i32m2(_src, _mult, vl);
    _shift = vrsub_vx_i32m2(_shift, -1, vl);
    _mulh = vssra_vv_i32m2(_mulh, vreinterpret_v_i32m2_u32m2(_shift), vl);
    _mulh = vadd_vx_i32m2(_mulh, out_zp, vl);
    vint16m1_t _tmp1 = vnclip_wx_i16m1(_mulh, 0, vl);
    vint8mf2_t _tmp2 = vnclip_wx_i8mf2(_tmp1, 0, vl);
    return _tmp2;
}
#elif defined RVV_0_7_1
static vint8m1_t requantize_m4_s(vint32m4_t _src, int32_t *multiplier, int32_t *shift,
                                 int32_t out_zp, int vl)
{
    vint32m4_t _mult = vle32_v_i32m4(multiplier, vl);
    vint32m4_t _shift = vle32_v_i32m4(shift, vl);
    vint32m4_t _mulh = vmulh_vv_i32m4(_src, _mult, vl);
    _shift = vrsub_vx_i32m4(_shift, -1, vl);
    _mulh = vssra_vv_i32m4(_mulh, vreinterpret_v_i32m4_u32m4(_shift), vl);
    _mulh = vadd_vx_i32m4(_mulh, out_zp, vl);
    vint16m2_t _tmp1 = vnclip_wx_i16m2(_mulh, 0, vl);
    vint8m1_t _tmp2 = vnclip_wx_i8m1(_tmp1, 0, vl);
    return _tmp2;
}
#endif

static inline void wg_b4f3s1_trans_input_packn_int8(const int8_t *src, int16_t *dst, int ch, int h,
                                                    int w, int blk_h, int blk_w, int8_t input_zp)
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
    [0] =  4 * r00 - 5 * r02 + r04
    [1] = -4 * (r01 + r02) + r04 + r03
    [2] =  4 * (r01 - r02) + r04 - r03
    [3] = -2 * (r01 - r03) + r04 - r02
    [4] =  2 * (r01 - r03) + r04 - r02
    [5] =  4 * r01 - 5 * r03 + r05
    */
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);
    int tiles = blk_h * blk_w;
    for (int q = 0; q + packn - 1 < ch; q += packn) {
        const int8_t *img0 = src + q * h * w;     // feature map after padding - q channel
        int16_t *img0_tm = dst + q * 36 * tiles;  // transform and interleave - q channel
        int16_t tmp[6][6][packn];
        for (int i = 0; i < blk_h; i++) {
            for (int j = 0; j < blk_w; j++) {
                // feature map after padding 6*6 start addr
                const int8_t *r0 = img0 + (i * w * 4 + j * 4) * packn;
                // input_tm1 6*6 block start addr
                int16_t *r0_tm = img0_tm + (i * blk_w + j) * packn;
                for (int m = 0; m < 6; m++) {
#ifdef RVV_1_0_0
                    vint8mf2_t _t00 = vle8_v_i8mf2(r0, vl);
                    vint8mf2_t _t01 = vle8_v_i8mf2(r0 + packn * 1, vl);
                    vint8mf2_t _t02 = vle8_v_i8mf2(r0 + packn * 2, vl);
                    vint8mf2_t _t03 = vle8_v_i8mf2(r0 + packn * 3, vl);
                    vint8mf2_t _t04 = vle8_v_i8mf2(r0 + packn * 4, vl);
                    vint8mf2_t _t05 = vle8_v_i8mf2(r0 + packn * 5, vl);
                    // (q - z)
                    vint16m1_t _r00 = vwsub_vx_i16m1(_t00, input_zp, vl);
                    vint16m1_t _r01 = vwsub_vx_i16m1(_t01, input_zp, vl);
                    vint16m1_t _r02 = vwsub_vx_i16m1(_t02, input_zp, vl);
                    vint16m1_t _r03 = vwsub_vx_i16m1(_t03, input_zp, vl);
                    vint16m1_t _r04 = vwsub_vx_i16m1(_t04, input_zp, vl);
                    vint16m1_t _r05 = vwsub_vx_i16m1(_t05, input_zp, vl);
                    vint16m1_t _tmp0m = vadd_vv_i16m1(
                        vadd_vv_i16m1(vmul_vx_i16m1(_r00, 4, vl), vmul_vx_i16m1(_r02, -5, vl), vl),
                        _r04, vl);
                    vint16m1_t _tmp1m = vmacc_vx_i16m1(vadd_vv_i16m1(_r04, _r03, vl), -4,
                                                       vadd_vv_i16m1(_r01, _r02, vl), vl);
                    vint16m1_t _tmp2m = vmacc_vx_i16m1(vsub_vv_i16m1(_r04, _r03, vl), 4,
                                                       vsub_vv_i16m1(_r01, _r02, vl), vl);
                    vint16m1_t _tmp3m = vmacc_vx_i16m1(vsub_vv_i16m1(_r04, _r02, vl), -2,
                                                       vsub_vv_i16m1(_r01, _r03, vl), vl);
                    vint16m1_t _tmp4m = vmacc_vx_i16m1(vsub_vv_i16m1(_r04, _r02, vl), 2,
                                                       vsub_vv_i16m1(_r01, _r03, vl), vl);
                    vint16m1_t _tmp5m = vadd_vv_i16m1(
                        vadd_vv_i16m1(vmul_vx_i16m1(_r01, 4, vl), vmul_vx_i16m1(_r03, -5, vl), vl),
                        _r05, vl);
                    // vint16m1_t _tmp0m = vwadd_wv_i16m1(vadd_vv_i16m1(vwmul_vx_i16m1(_r00, 4, vl),
                    // vwmul_vx_i16m1(_r02, -5, vl), vl), _r04, vl); vint16m1_t _tmp1m =
                    // vmacc_vx_i16m1(vwadd_vv_i16m1(_r04, _r03, vl), -4, vwadd_vv_i16m1(_r01, _r02,
                    // vl), vl); vint16m1_t _tmp2m = vmacc_vx_i16m1(vwsub_vv_i16m1(_r04, _r03, vl),
                    // 4, vwsub_vv_i16m1(_r01, _r02, vl), vl); vint16m1_t _tmp3m =
                    // vmacc_vx_i16m1(vwsub_vv_i16m1(_r04, _r02, vl), -2, vwsub_vv_i16m1(_r01, _r03,
                    // vl), vl); vint16m1_t _tmp4m = vmacc_vx_i16m1(vwsub_vv_i16m1(_r04, _r02, vl),
                    // 2, vwsub_vv_i16m1(_r01, _r03, vl), vl); vint16m1_t _tmp5m =
                    // vwadd_wv_i16m1(vadd_vv_i16m1(vwmul_vx_i16m1(_r01, 4, vl),
                    // vwmul_vx_i16m1(_r03, -5, vl), vl), _r05, vl);
                    vse16_v_i16m1(tmp[0][m], _tmp0m, vl);
                    vse16_v_i16m1(tmp[1][m], _tmp1m, vl);
                    vse16_v_i16m1(tmp[2][m], _tmp2m, vl);
                    vse16_v_i16m1(tmp[3][m], _tmp3m, vl);
                    vse16_v_i16m1(tmp[4][m], _tmp4m, vl);
                    vse16_v_i16m1(tmp[5][m], _tmp5m, vl);
#elif defined RVV_0_7_1
                    vint8m1_t _t00 = vle8_v_i8m1(r0, vl);
                    vint8m1_t _t01 = vle8_v_i8m1(r0 + packn * 1, vl);
                    vint8m1_t _t02 = vle8_v_i8m1(r0 + packn * 2, vl);
                    vint8m1_t _t03 = vle8_v_i8m1(r0 + packn * 3, vl);
                    vint8m1_t _t04 = vle8_v_i8m1(r0 + packn * 4, vl);
                    vint8m1_t _t05 = vle8_v_i8m1(r0 + packn * 5, vl);
                    // (q - z)
                    vint16m2_t _r00 = vwsub_vx_i16m2(_t00, input_zp, vl);
                    vint16m2_t _r01 = vwsub_vx_i16m2(_t01, input_zp, vl);
                    vint16m2_t _r02 = vwsub_vx_i16m2(_t02, input_zp, vl);
                    vint16m2_t _r03 = vwsub_vx_i16m2(_t03, input_zp, vl);
                    vint16m2_t _r04 = vwsub_vx_i16m2(_t04, input_zp, vl);
                    vint16m2_t _r05 = vwsub_vx_i16m2(_t05, input_zp, vl);
                    vint16m2_t _tmp0m = vadd_vv_i16m2(
                        vadd_vv_i16m2(vmul_vx_i16m2(_r00, 4, vl), vmul_vx_i16m2(_r02, -5, vl), vl),
                        _r04, vl);
                    vint16m2_t _tmp1m = vmacc_vx_i16m2(vadd_vv_i16m2(_r04, _r03, vl), -4,
                                                       vadd_vv_i16m2(_r01, _r02, vl), vl);
                    vint16m2_t _tmp2m = vmacc_vx_i16m2(vsub_vv_i16m2(_r04, _r03, vl), 4,
                                                       vsub_vv_i16m2(_r01, _r02, vl), vl);
                    vint16m2_t _tmp3m = vmacc_vx_i16m2(vsub_vv_i16m2(_r04, _r02, vl), -2,
                                                       vsub_vv_i16m2(_r01, _r03, vl), vl);
                    vint16m2_t _tmp4m = vmacc_vx_i16m2(vsub_vv_i16m2(_r04, _r02, vl), 2,
                                                       vsub_vv_i16m2(_r01, _r03, vl), vl);
                    vint16m2_t _tmp5m = vadd_vv_i16m2(
                        vadd_vv_i16m2(vmul_vx_i16m2(_r01, 4, vl), vmul_vx_i16m2(_r03, -5, vl), vl),
                        _r05, vl);
                    // vint16m1_t _tmp0m = vwadd_wv_i16m1(vadd_vv_i16m1(vwmul_vx_i16m1(_r00, 4, vl),
                    // vwmul_vx_i16m1(_r02, -5, vl), vl), _r04, vl); vint16m1_t _tmp1m =
                    // vmacc_vx_i16m1(vwadd_vv_i16m1(_r04, _r03, vl), -4, vwadd_vv_i16m1(_r01, _r02,
                    // vl), vl); vint16m1_t _tmp2m = vmacc_vx_i16m1(vwsub_vv_i16m1(_r04, _r03, vl),
                    // 4, vwsub_vv_i16m1(_r01, _r02, vl), vl); vint16m1_t _tmp3m =
                    // vmacc_vx_i16m1(vwsub_vv_i16m1(_r04, _r02, vl), -2, vwsub_vv_i16m1(_r01, _r03,
                    // vl), vl); vint16m1_t _tmp4m = vmacc_vx_i16m1(vwsub_vv_i16m1(_r04, _r02, vl),
                    // 2, vwsub_vv_i16m1(_r01, _r03, vl), vl); vint16m1_t _tmp5m =
                    // vwadd_wv_i16m1(vadd_vv_i16m1(vwmul_vx_i16m1(_r01, 4, vl),
                    // vwmul_vx_i16m1(_r03, -5, vl), vl), _r05, vl);
                    vse16_v_i16m2(tmp[0][m], _tmp0m, vl);
                    vse16_v_i16m2(tmp[1][m], _tmp1m, vl);
                    vse16_v_i16m2(tmp[2][m], _tmp2m, vl);
                    vse16_v_i16m2(tmp[3][m], _tmp3m, vl);
                    vse16_v_i16m2(tmp[4][m], _tmp4m, vl);
                    vse16_v_i16m2(tmp[5][m], _tmp5m, vl);
#endif
                    r0 += w * packn;
                }
                for (int m = 0; m < 6; m++) {
                    int16_t *r0_tm0 = r0_tm;
                    int16_t *r0_tm1 = r0_tm0 + tiles * packn;
                    int16_t *r0_tm2 = r0_tm1 + tiles * packn;
                    int16_t *r0_tm3 = r0_tm2 + tiles * packn;
                    int16_t *r0_tm4 = r0_tm3 + tiles * packn;
                    int16_t *r0_tm5 = r0_tm4 + tiles * packn;
                    vint16m1_t _tmp00 = vle16_v_i16m1(tmp[m][0], vl);
                    vint16m1_t _tmp01 = vle16_v_i16m1(tmp[m][1], vl);
                    vint16m1_t _tmp02 = vle16_v_i16m1(tmp[m][2], vl);
                    vint16m1_t _tmp03 = vle16_v_i16m1(tmp[m][3], vl);
                    vint16m1_t _tmp04 = vle16_v_i16m1(tmp[m][4], vl);
                    vint16m1_t _tmp05 = vle16_v_i16m1(tmp[m][5], vl);
                    vint16m1_t _r0tm0 =
                        vmacc_vx_i16m1(vmacc_vx_i16m1(_tmp04, 4, _tmp00, vl), -5, _tmp02, vl);
                    vint16m1_t _r0tm1 = vmacc_vx_i16m1(vadd_vv_i16m1(_tmp04, _tmp03, vl), -4,
                                                       vadd_vv_i16m1(_tmp01, _tmp02, vl), vl);
                    vint16m1_t _r0tm2 = vmacc_vx_i16m1(vsub_vv_i16m1(_tmp04, _tmp03, vl), 4,
                                                       vsub_vv_i16m1(_tmp01, _tmp02, vl), vl);
                    vint16m1_t _r0tm3 = vmacc_vx_i16m1(vsub_vv_i16m1(_tmp04, _tmp02, vl), -2,
                                                       vsub_vv_i16m1(_tmp01, _tmp03, vl), vl);
                    vint16m1_t _r0tm4 = vmacc_vx_i16m1(vsub_vv_i16m1(_tmp04, _tmp02, vl), 2,
                                                       vsub_vv_i16m1(_tmp01, _tmp03, vl), vl);
                    vint16m1_t _r0tm5 =
                        vmacc_vx_i16m1(vmacc_vx_i16m1(_tmp05, 4, _tmp01, vl), -5, _tmp03, vl);
                    vse16_v_i16m1(r0_tm0, _r0tm0, vl);
                    vse16_v_i16m1(r0_tm1, _r0tm1, vl);
                    vse16_v_i16m1(r0_tm2, _r0tm2, vl);
                    vse16_v_i16m1(r0_tm3, _r0tm3, vl);
                    vse16_v_i16m1(r0_tm4, _r0tm4, vl);
                    vse16_v_i16m1(r0_tm5, _r0tm5, vl);
                    r0_tm += tiles * packn * 6;
                }
            }
        }
    }
}

static inline void wg_b4f3s1_trans_output_packn_int8(const int32_t *src, const int32_t *bias,
                                                     int8_t *dst, int ch, int blk_h, int blk_w,
                                                     int32_t *multi, int32_t *shift, int32_t out_zp)
{
    /* output transform matrix
    AT = {
        { 1  1  1   1  1   0 },
        { 0  1  -1  2  -2  0 },
        { 0  1  1   4  4   0 },
        { 0  1  -1  8  -8  1 }
    };
    AT = {
        { 1  1  1   1  1   0 },
        { 0  1  -1  2  -2  0 },
        { 0  1  1   4  4   0 },
        { 0  1  -1  8  -8  4 }  // 和 G 变换矩阵一起将累加和扩大了 24 * 24 倍
    };
    [0] = r00 + (r01 + r02) + (r03 + r04)
    [1] =       (r01 - r02) + (r03 - r04) * 2
    [2] =       (r01 + r02) + (r03 + r04) * 4
    [3] = 4 * r05 + (r01 - r02) + (r03 - r04) * 8
    */
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);
    int tiles = blk_h * blk_w;
    for (int p = 0; p + packn - 1 < ch; p += packn) {
        const int32_t *out0_tm = src + p * 36 * tiles;   // 输出转换前/dot后 第p个channel
        int8_t *out0 = dst + p * 4 * blk_h * 4 * blk_w;  // 转换后输出 第p个channel
        int32_t tmp[4][6][packn];

        vint32m2_t _bias = bias ? vle32_v_i32m2(bias + p, vl) : vmv_v_x_i32m2(0, vl);
        _bias = vmul_vx_i32m2(_bias, 576, vl);
        for (int i = 0; i < blk_h; i++) {
            for (int j = 0; j < blk_w; j++) {
                const int32_t *output0_tm_0 = out0_tm + (i * blk_w + j) * packn;  // 6*6 起始地址
                const int32_t *output0_tm_1 = output0_tm_0 + tiles * packn * 1;
                const int32_t *output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const int32_t *output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                const int32_t *output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                const int32_t *output0_tm_5 = output0_tm_0 + tiles * packn * 5;
                int8_t *output0 = out0 + (i * blk_w * 4 * 4 + j * 4) * packn;  // out 4*4 addr
                for (int m = 0; m < 6; m++) {
                    vint32m2_t _r00 = vle32_v_i32m2(output0_tm_0, vl);
                    vint32m2_t _r01 = vle32_v_i32m2(output0_tm_1, vl);
                    vint32m2_t _r02 = vle32_v_i32m2(output0_tm_2, vl);
                    vint32m2_t _r03 = vle32_v_i32m2(output0_tm_3, vl);
                    vint32m2_t _r04 = vle32_v_i32m2(output0_tm_4, vl);
                    vint32m2_t _r05 = vle32_v_i32m2(output0_tm_5, vl);
                    vint32m2_t _tmp02a = vadd_vv_i32m2(_r01, _r02, vl);
                    vint32m2_t _tmp13a = vsub_vv_i32m2(_r01, _r02, vl);
                    vint32m2_t _tmp02b = vadd_vv_i32m2(_r03, _r04, vl);
                    vint32m2_t _tmp13b = vsub_vv_i32m2(_r03, _r04, vl);
                    vint32m2_t _tmp0m =
                        vadd_vv_i32m2(vadd_vv_i32m2(_r00, _tmp02a, vl), _tmp02b, vl);
                    vint32m2_t _tmp1m = vmacc_vx_i32m2(_tmp13a, 2, _tmp13b, vl);
                    vint32m2_t _tmp2m = vmacc_vx_i32m2(_tmp02a, 4, _tmp02b, vl);
                    vint32m2_t _tmp3m =
                        vmacc_vx_i32m2(vmacc_vx_i32m2(_tmp13a, 4, _r05, vl), 8, _tmp13b, vl);
                    vse32_v_i32m2(tmp[0][m], _tmp0m, vl);
                    vse32_v_i32m2(tmp[1][m], _tmp1m, vl);
                    vse32_v_i32m2(tmp[2][m], _tmp2m, vl);
                    vse32_v_i32m2(tmp[3][m], _tmp3m, vl);
                    output0_tm_0 += tiles * packn * 6;
                    output0_tm_1 += tiles * packn * 6;
                    output0_tm_2 += tiles * packn * 6;
                    output0_tm_3 += tiles * packn * 6;
                    output0_tm_4 += tiles * packn * 6;
                    output0_tm_5 += tiles * packn * 6;
                }
                for (int m = 0; m < 4; m++) {
                    vint32m2_t _tmp00 = vle32_v_i32m2(tmp[m][0], vl);
                    vint32m2_t _tmp01 = vle32_v_i32m2(tmp[m][1], vl);
                    vint32m2_t _tmp02 = vle32_v_i32m2(tmp[m][2], vl);
                    vint32m2_t _tmp03 = vle32_v_i32m2(tmp[m][3], vl);
                    vint32m2_t _tmp04 = vle32_v_i32m2(tmp[m][4], vl);
                    vint32m2_t _tmp05 = vle32_v_i32m2(tmp[m][5], vl);
                    vint32m2_t _tmp02a = vadd_vv_i32m2(_tmp01, _tmp02, vl);
                    vint32m2_t _tmp13a = vsub_vv_i32m2(_tmp01, _tmp02, vl);
                    vint32m2_t _tmp02b = vadd_vv_i32m2(_tmp03, _tmp04, vl);
                    vint32m2_t _tmp13b = vsub_vv_i32m2(_tmp03, _tmp04, vl);
                    vint32m2_t _out00 = vadd_vv_i32m2(
                        _bias, vadd_vv_i32m2(vadd_vv_i32m2(_tmp00, _tmp02a, vl), _tmp02b, vl), vl);
                    vint32m2_t _out01 =
                        vadd_vv_i32m2(_bias, vmacc_vx_i32m2(_tmp13a, 2, _tmp13b, vl), vl);
                    vint32m2_t _out02 =
                        vadd_vv_i32m2(_bias, vmacc_vx_i32m2(_tmp02a, 4, _tmp02b, vl), vl);
                    vint32m2_t _out03 = vadd_vv_i32m2(
                        _bias,
                        vmacc_vx_i32m2(vmacc_vx_i32m2(_tmp13a, 4, _tmp05, vl), 8, _tmp13b, vl), vl);
#ifdef RVV_1_0_0
                    vint8mf2_t _res0 = requantize_m2_s(_out00, multi + p, shift + p, out_zp, vl);
                    vint8mf2_t _res1 = requantize_m2_s(_out01, multi + p, shift + p, out_zp, vl);
                    vint8mf2_t _res2 = requantize_m2_s(_out02, multi + p, shift + p, out_zp, vl);
                    vint8mf2_t _res3 = requantize_m2_s(_out03, multi + p, shift + p, out_zp, vl);
                    vse8_v_i8mf2(output0, _res0, vl);
                    vse8_v_i8mf2(output0 + packn * 1, _res1, vl);
                    vse8_v_i8mf2(output0 + packn * 2, _res2, vl);
                    vse8_v_i8mf2(output0 + packn * 3, _res3, vl);
#elif defined RVV_0_7_1
                    // vint8m1_t _res0 = requantize_m4_s(vlmul_ext_v_i32m2_i32m4(_out00), multi + p,
                    //                                   shift + p, out_zp, vl);
                    // vint8m1_t _res1 = requantize_m4_s(vlmul_ext_v_i32m2_i32m4(_out01), multi + p,
                    //                                   shift + p, out_zp, vl);
                    // vint8m1_t _res2 = requantize_m4_s(vlmul_ext_v_i32m2_i32m4(_out02), multi + p,
                    //                                   shift + p, out_zp, vl);
                    // vint8m1_t _res3 = requantize_m4_s(vlmul_ext_v_i32m2_i32m4(_out03), multi + p,
                    //                                   shift + p, out_zp, vl);
                    vint8m1_t _res0 =
                        requantize_m4_s(vset_v_i32m2_i32m4(vundefined_i32m4(), 0, _out00),
                                        multi + p, shift + p, out_zp, vl);
                    vint8m1_t _res1 =
                        requantize_m4_s(vset_v_i32m2_i32m4(vundefined_i32m4(), 0, _out01),
                                        multi + p, shift + p, out_zp, vl);
                    vint8m1_t _res2 =
                        requantize_m4_s(vset_v_i32m2_i32m4(vundefined_i32m4(), 0, _out02),
                                        multi + p, shift + p, out_zp, vl);
                    vint8m1_t _res3 =
                        requantize_m4_s(vset_v_i32m2_i32m4(vundefined_i32m4(), 0, _out03),
                                        multi + p, shift + p, out_zp, vl);
                    vse8_v_i8m1(output0, _res0, vl);
                    vse8_v_i8m1(output0 + packn * 1, _res1, vl);
                    vse8_v_i8m1(output0 + packn * 2, _res2, vl);
                    vse8_v_i8m1(output0 + packn * 3, _res3, vl);
#endif
                    output0 += blk_w * 4 * packn;
                }
            }
        }
    }
}

static inline void wg_bxf3s1_reorder_input_tile8_int8(const int16_t *src, int16_t *dst, int ch,
                                                      int tiles, int area)
{
    const int packn = csrr_vlenb() / sizeof(int16_t);
    const int vl = vsetvl_e16m1(packn);
    for (int r = 0; r < area; r++) {
        int16_t *img_tm2 = dst + r * tiles * ch;  // input_tm2 r channel data
        int t = 0;
        for (; t + 7 < tiles; t += 8) {
            const int16_t *tm1 = src;
            tm1 += (r * tiles + t) * packn;
            for (int q = 0; q < ch / packn; q++) {
                vint16m1_t _tmp0 = vle16_v_i16m1(tm1, vl);
                vint16m1_t _tmp1 = vle16_v_i16m1(tm1 + packn * 1, vl);
                vint16m1_t _tmp2 = vle16_v_i16m1(tm1 + packn * 2, vl);
                vint16m1_t _tmp3 = vle16_v_i16m1(tm1 + packn * 3, vl);
                vint16m1_t _tmp4 = vle16_v_i16m1(tm1 + packn * 4, vl);
                vint16m1_t _tmp5 = vle16_v_i16m1(tm1 + packn * 5, vl);
                vint16m1_t _tmp6 = vle16_v_i16m1(tm1 + packn * 6, vl);
                vint16m1_t _tmp7 = vle16_v_i16m1(tm1 + packn * 7, vl);
                vsseg8e16_v_i16m1(img_tm2, _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7,
                                  vl);
                tm1 += area * tiles * packn;
                img_tm2 += 8 * packn;
            }
        }
        for (; t + 3 < tiles; t += 4) {
            const int16_t *tm1 = src;
            tm1 += (r * tiles + t) * packn;
            for (int q = 0; q < ch / packn; q++) {
                vint16m1_t _tmp0 = vle16_v_i16m1(tm1, vl);
                vint16m1_t _tmp1 = vle16_v_i16m1(tm1 + packn * 1, vl);
                vint16m1_t _tmp2 = vle16_v_i16m1(tm1 + packn * 2, vl);
                vint16m1_t _tmp3 = vle16_v_i16m1(tm1 + packn * 3, vl);
                vsseg4e16_v_i16m1(img_tm2, _tmp0, _tmp1, _tmp2, _tmp3, vl);
                tm1 += area * tiles * packn;
                img_tm2 += 4 * packn;
            }
        }
        for (; t + 1 < tiles; t += 2) {
            const int16_t *tm1 = src;
            tm1 += (r * tiles + t) * packn;
            for (int q = 0; q < ch / packn; q++) {
                vint16m1_t _tmp0 = vle16_v_i16m1(tm1, vl);
                vint16m1_t _tmp1 = vle16_v_i16m1(tm1 + packn * 1, vl);
                vsseg2e16_v_i16m1(img_tm2, _tmp0, _tmp1, vl);
                tm1 += area * tiles * packn;
                img_tm2 += 2 * packn;
            }
        }
        for (; t < tiles; t++) {
            const int16_t *tm1 = src;
            tm1 += (r * tiles + t) * packn;
            for (int q = 0; q < ch / packn; q++) {
                vint16m1_t _tmp0 = vle16_v_i16m1(tm1, vl);
                vse16_v_i16m1(img_tm2, _tmp0, vl);
                tm1 += area * tiles * packn;
                img_tm2 += 1 * packn;
            }
        }
    }
}

static inline void wg_bxf3s1_batch_gemm_m8n8_int8(const int16_t *input, const int16_t *kernel,
                                                  int32_t *output, int in_ch, int out_ch, int tiles,
                                                  int area)
{
    const int packn = csrr_vlenb() / sizeof(int16_t);
    const int vl = vsetvl_e16m1(packn);

    for (int p = 0; p + packn - 1 < out_ch; p += packn) {
        int32_t *output0_tm = output + p * area * tiles;        // 8 channel dot output
        const int16_t *kernel0_tm = kernel + p * area * in_ch;  // 8 channel kernel
        for (int r = 0; r < area; r++) {
            const int16_t *img0 = input + r * tiles * in_ch;  // img_tm2 第r个channel
            int t = 0;
            for (; t + 7 < tiles; t += 8) {
                const int16_t *k0 = kernel0_tm + r * in_ch * packn;

                vint32m2_t _acc00 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _acc01 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _acc02 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _acc03 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _acc04 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _acc05 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _acc06 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _acc07 = vmv_v_x_i32m2(0, vl);

                for (int c = 0; c < in_ch; c++) {
                    vint16m1_t _kernel0 = vle16_v_i16m1(k0, vl);
                    k0 += packn;
                    _acc00 = vwmacc_vx_i32m2(_acc00, img0[0], _kernel0, vl);
                    _acc01 = vwmacc_vx_i32m2(_acc01, img0[1], _kernel0, vl);
                    _acc02 = vwmacc_vx_i32m2(_acc02, img0[2], _kernel0, vl);
                    _acc03 = vwmacc_vx_i32m2(_acc03, img0[3], _kernel0, vl);
                    _acc04 = vwmacc_vx_i32m2(_acc04, img0[4], _kernel0, vl);
                    _acc05 = vwmacc_vx_i32m2(_acc05, img0[5], _kernel0, vl);
                    _acc06 = vwmacc_vx_i32m2(_acc06, img0[6], _kernel0, vl);
                    _acc07 = vwmacc_vx_i32m2(_acc07, img0[7], _kernel0, vl);
                    img0 += 8;
                }
                vse32_v_i32m2(output0_tm, _acc00, vl);
                vse32_v_i32m2(output0_tm + packn * 1, _acc01, vl);
                vse32_v_i32m2(output0_tm + packn * 2, _acc02, vl);
                vse32_v_i32m2(output0_tm + packn * 3, _acc03, vl);
                vse32_v_i32m2(output0_tm + packn * 4, _acc04, vl);
                vse32_v_i32m2(output0_tm + packn * 5, _acc05, vl);
                vse32_v_i32m2(output0_tm + packn * 6, _acc06, vl);
                vse32_v_i32m2(output0_tm + packn * 7, _acc07, vl);
                output0_tm += packn * 8;
            }
            for (; t + 3 < tiles; t += 4) {
                const int16_t *k0 = kernel0_tm + r * in_ch * packn;

                vint32m2_t _acc00 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _acc01 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _acc02 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _acc03 = vmv_v_x_i32m2(0, vl);

                for (int c = 0; c < in_ch; c++) {
                    vint16m1_t _kernel0 = vle16_v_i16m1(k0, vl);
                    k0 += packn;
                    _acc00 = vwmacc_vx_i32m2(_acc00, img0[0], _kernel0, vl);
                    _acc01 = vwmacc_vx_i32m2(_acc01, img0[1], _kernel0, vl);
                    _acc02 = vwmacc_vx_i32m2(_acc02, img0[2], _kernel0, vl);
                    _acc03 = vwmacc_vx_i32m2(_acc03, img0[3], _kernel0, vl);
                    img0 += 4;
                }
                vse32_v_i32m2(output0_tm, _acc00, vl);
                vse32_v_i32m2(output0_tm + packn * 1, _acc01, vl);
                vse32_v_i32m2(output0_tm + packn * 2, _acc02, vl);
                vse32_v_i32m2(output0_tm + packn * 3, _acc03, vl);
                output0_tm += packn * 4;
            }
            for (; t + 1 < tiles; t += 2) {
                const int16_t *k0 = kernel0_tm + r * in_ch * packn;

                vint32m2_t _acc00 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _acc01 = vmv_v_x_i32m2(0, vl);

                for (int c = 0; c < in_ch; c++) {
                    vint16m1_t _kernel0 = vle16_v_i16m1(k0, vl);
                    k0 += packn;
                    _acc00 = vwmacc_vx_i32m2(_acc00, img0[0], _kernel0, vl);
                    _acc01 = vwmacc_vx_i32m2(_acc01, img0[1], _kernel0, vl);
                    img0 += 2;
                }
                vse32_v_i32m2(output0_tm, _acc00, vl);
                vse32_v_i32m2(output0_tm + packn * 1, _acc01, vl);
                output0_tm += packn * 2;
            }
            for (; t < tiles; t++) {
                const int16_t *k0 = kernel0_tm + r * in_ch * packn;

                vint32m2_t _acc00 = vmv_v_x_i32m2(0, vl);

                for (int c = 0; c < in_ch; c++) {
                    vint16m1_t _kernel0 = vle16_v_i16m1(k0, vl);
                    k0 += packn;
                    _acc00 = vwmacc_vx_i32m2(_acc00, img0[0], _kernel0, vl);
                    img0 += 1;
                }
                vse32_v_i32m2(output0_tm, _acc00, vl);
                output0_tm += packn * 1;
            }
        }
    }
}

/******************************************************************************************
 * kernel layout before:  [O, I, 3, 3]
 * kernel layout after :  [O/packn, 36, I, packn]
 * constrain: output channel % packn = 0
 *            input channel % packn = 0
 ******************************************************************************************/
void shl_rvv_wg_b4f3s1_trans_kernel_packn_int8(struct csinn_tensor *src_kernel,
                                               struct csinn_tensor *dst_kernel)
{
    int32_t outch = src_kernel->dim[0];
    int32_t inch = src_kernel->dim[1];
    int8_t *kernel_data = (int8_t *)src_kernel->data;
    // for kernel transform buf, 3x3 --> 6x6
    int16_t *kernel_tm = (int16_t *)shl_mem_alloc(outch * inch * 6 * 6 * sizeof(int16_t));
    // kernel transform matrix: G
    const int16_t ktm[6][3] = {{6, 0, 0}, {-4, -4, -4}, {-4, 4, -4},
                               {1, 2, 4}, {1, -2, 4},   {0, 0, 6}};
    csinn_tensor_copy(dst_kernel, src_kernel);  // tensor->dtype ??
    for (int p = 0; p < outch; p++) {
        for (int q = 0; q < inch; q++) {
            const int8_t *kernel0 = kernel_data + p * inch * 9 + q * 9;
            int16_t *kernel_tm0 = kernel_tm + p * inch * 36 + q * 36;
            // transform kernel
            const int8_t *k0 = kernel0;
            const int8_t *k1 = kernel0 + 3;
            const int8_t *k2 = kernel0 + 6;
            // h : first compute the transport matrix tmp = (g * GT)T
            int16_t tmp[6][3];
            for (int i = 0; i < 6; i++) {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }
            // U
            for (int j = 0; j < 6; j++) {
                int16_t *tmpp = &tmp[j][0];
                for (int i = 0; i < 6; i++) {
                    kernel_tm0[j * 6 + i] =
                        tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
    // optimized layout for winograd b4f3
    // [O, I, 6, 6]  -->  [O/packn, 6*6, I, packn]
    int16_t *kernel_tm_packn =
        (int16_t *)shl_mem_alloc(outch / 8 * 36 * inch * 8 * sizeof(int16_t));
    dst_kernel->data = kernel_tm_packn;

    const int packn = csrr_vlenb() / sizeof(int16_t);
    for (int oc = 0; oc + packn - 1 < outch; oc += packn) {
        int16_t *g0 = kernel_tm_packn + oc * 36 * inch;
        for (int k = 0; k < 36; k++) {
            int16_t *g00 = g0 + k * inch * packn;
            for (int ic = 0; ic < inch; ic++) {
                for (int j = 0; j < packn; j++) {
                    int16_t *k00 = kernel_tm + (oc + j) * 36 * inch + ic * 36;
                    *g00++ = k00[k];
                }
            }
        }
    }
    shl_mem_free(kernel_tm);
}

/******************************************************************************************
 * constrain: output channel % 8 = 0
 *            input channel % 8 = 0
 ******************************************************************************************/
int shl_rvv_wg_b4f3s1_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NCHW) {
        shl_rvv_tensor_ndarray_to_nc1xc0_replace_int8(input);
    }
    if (output->layout == CSINN_LAYOUT_NCHW) {
        output->dim[1] /= input->dim[4];
        output->dim[4] = input->dim[4];
        output->dim_count = 5;
        output->layout = CSINN_LAYOUT_NC1HWC0;
    }
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int16_t *kernel_data = (int16_t *)params->conv_extra.kernel_tm->data;
    int32_t *bias_data = (int32_t *)bias->data;
    // param
    int pad_left = params->pad_left;
    int pad_top = params->pad_top;
    int batch = input->dim[0];
    int in_c = input->dim[1] * input->dim[4];
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
        int8_t *input_padd_buf = (int8_t *)shl_mem_alloc(in_c * padded_in_hw * sizeof(int8_t));
        // pad input
        winograd_pad_input_packn_int8(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h,
                                      padded_in_w, pad_top, pad_left, input->qinfo->zero_point);
        input_data += input_size;
        /****************************** transform input *****************************/
        // input transform buffer1: [in_ch/8, 64, tiles, 8]
        int16_t *input_tm1_buf =
            (int16_t *)shl_mem_alloc(in_c / 8 * 36 * tiles * 8 * sizeof(int16_t));
        wg_b4f3s1_trans_input_packn_int8(input_padd_buf, input_tm1_buf, in_c, padded_in_h,
                                         padded_in_w, block_h, block_w, input->qinfo->zero_point);
        shl_mem_free(input_padd_buf);
        /****************************** reorder input_tm1_buf *****************************/
        // input reorder buffer2: [36, tiles/8, in_c, 8]
        int16_t *input_tm2_buf = (int16_t *)shl_mem_alloc(36 * tiles * in_c * sizeof(int16_t));
        wg_bxf3s1_reorder_input_tile8_int8(input_tm1_buf, input_tm2_buf, in_c, tiles, 36);
        shl_mem_free(input_tm1_buf);
        /****************************** batch gemm *****************************/
        // output_dot_buf： [out_c/8, 36, tiles, 8]
        const int vlen = csrr_vlenb() * 8;
        int32_t *output_dot_buf =
            (int32_t *)shl_mem_alloc(out_c / 8 * 36 * tiles * 8 * sizeof(int32_t));

        wg_bxf3s1_batch_gemm_m8n8_int8(input_tm2_buf, kernel_data, output_dot_buf, in_c, out_c,
                                       tiles, 36);

        shl_mem_free(input_tm2_buf);
        /****************************** transform output *****************************/
        // output_tm1_buf: [out_c/8, out_h4, out_w4, 8]
        int8_t *output_tm1_buf =
            (int8_t *)shl_mem_alloc(out_c / 8 * tiles * 4 * 4 * 8 * sizeof(int8_t));
        int32_t *multiplier = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));
        int32_t *shift = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));
        if (kernel->quant_channel > 1) {
            for (int c = 0; c < out_c; c++) {
                multiplier[c] = kernel->qinfo[c].multiplier;
                shift[c] = kernel->qinfo[c].shift;
            }
        } else if (kernel->quant_channel == 1) {
            for (int c = 0; c < out_c; c++) {
                multiplier[c] = kernel->qinfo[0].multiplier;
                shift[c] = kernel->qinfo[0].shift;
            }
        }
        wg_b4f3s1_trans_output_packn_int8(output_dot_buf, bias_data, output_tm1_buf, out_c, block_h,
                                          block_w, multiplier, shift, output->qinfo->zero_point);
        shl_mem_free(output_dot_buf);
        // crop the output after transform: cut extra part (right , bottom)
        winograd_crop_output_packn_int8(output_tm1_buf, output_data, out_c, out_h, out_w,
                                        block_h * 4, block_w * 4);
        output_data += output_size;
        shl_mem_free(output_tm1_buf);
        shl_mem_free(multiplier);
        shl_mem_free(shift);
    }
    return CSINN_TRUE;
}
