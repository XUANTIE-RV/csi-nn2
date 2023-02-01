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

/**************************************************************************************************

    void gemm_fp32_ncxhwx_12xpack2n(const float *output,
                                    const float *kernel,
                                    const float *input,
                                    const float *bias,
                                    int m,          // maxtrix A row
                                    int k,          // maxtrix A col / maxtrix B row
                                    int n,          // maxtrix B col
                                    bool fuse_relu)

    Algorithm works as follows:
        (1) perform matrix-multiplication [pack2n, k] x [k, n] = [pack2n, n]
            ...

 *************************************************************************************************/
    .file           "gemm_fp32_ncxhwx.S"
    .section        .text.gemm_fp32_ncxhwx_12xpack2n, "ax", @progbits
    .align          5
    .global         gemm_fp32_ncxhwx_12xpack2n
    .type           gemm_fp32_ncxhwx_12xpack2n, @function

gemm_fp32_ncxhwx_12xpack2n:
    // TODO: 8*pack2n or 12*pack2n ???
    ret
    .end

