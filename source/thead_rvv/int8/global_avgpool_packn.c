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
 * note: VLEN = 128/256 ... flexible vlen
 *************************************************************/

/* int8 --> fp16 acc --> int8 */
int shl_rvv_global_avgpool2d_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_pool_params *params)
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

    int batch = input->dim[0];
    int in_c = input->dim[1] * input->dim[4];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int in_hw = in_h * in_w;

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c + packn - 1 < in_c; c += packn) {
            vint8m1_t _input = vle8_v_i8m1(input_data, vl);
            input_data += packn;
            vint16m2_t _tmp = vwsub_vx_i16m2(_input, (int8_t)input->qinfo->zero_point, vl);
            vfloat16m2_t _acc =
                vfmul_vf_f16m2(vfcvt_f_x_v_f16m2(_tmp, vl), input->qinfo->scale, vl);
            for (int i = 1; i < in_hw; i++) {
                _tmp = vwsub_vx_i16m2(vle8_v_i8m1(input_data, vl), (int8_t)input->qinfo->zero_point,
                                      vl);
                vfloat16m2_t _inputf =
                    vfmul_vf_f16m2(vfcvt_f_x_v_f16m2(_tmp, vl), input->qinfo->scale, vl);
                _acc = vfadd_vv_f16m2(_acc, _inputf, vl);
                input_data += packn;
            }
            vfloat16m2_t _avg = vfmul_vf_f16m2(_acc, 1.0f / in_hw / output->qinfo->scale, vl);
            _avg = vfadd_vf_f16m2(_avg, output->qinfo->zero_point, vl);
            vint16m2_t _output = vfcvt_x_f_v_i16m2(_avg, vl);
            vint8m1_t _res = vnclip_wx_i8m1(_output, 0, vl);
            vse8_v_i8m1(output_data, _res, vl);
            output_data += packn;
        }
    }
    return CSINN_TRUE;
}
