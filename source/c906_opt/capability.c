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

#include "shl_c906.h"

static int common_all_support(struct csinn_tensor *input, struct csinn_params_base *base)
{
    /* only support layout: NCHW */
    if (base->layout != CSINN_LAYOUT_NCHW) {
        return CSINN_OPT_UNSUPPORTED;
    }

    if ((input->dtype != CSINN_DTYPE_FLOAT16) && (input->dtype != CSINN_DTYPE_FLOAT32)) {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_ASM;
}

int shl_c906_conv2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv2d_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c906_depthwise_conv2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params)
{
    /* only support layout: NCHW */
    if (params->base.layout != CSINN_LAYOUT_NCHW) {
        return CSINN_OPT_UNSUPPORTED;
    }
    int32_t kernel_h = kernel->dim[2];
    int32_t kernel_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
            return CSINN_OPT_ASM;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
            return CSINN_OPT_ASM;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
            return CSINN_OPT_ASM;
        } else if (kernel_h == 5 && kernel_w == 5 && stride_h == 1 && stride_w == 1) {
            return CSINN_OPT_ASM;
        } else if (kernel_h == 5 && kernel_w == 5 && stride_h == 2 && stride_w == 2) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_conv1d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv1d_params *params)
{
    int32_t kernel_w = kernel->dim[2];
    int32_t stride_w = params->stride_width;
    int32_t dalition_w = params->dilation_width;
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        if (kernel_w == 1 && stride_w == 1 && dalition_w == 1) {
            if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
                return CSINN_OPT_INTRINSIC;
            } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
                return CSINN_OPT_INTRINSIC;
            } else {
                return CSINN_OPT_C_REFERENCE;
            }
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        if (kernel_w == 1 && stride_w == 1 && dalition_w == 1) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_depthwise_conv1d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv1d_params *params)
{
    int32_t kernel_w = kernel->dim[2];
    int32_t stride_w = params->stride_width;

    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        if (kernel_w == 8 && stride_w == 1) {
            return CSINN_OPT_INTRINSIC;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_fullyconnected_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *weights, struct csinn_tensor *bias,
                                struct csinn_fc_params *params)
{
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        return CSINN_OPT_ASM;
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_maxpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params)
{
    /* only support layout: NCHW */
    if (params->base.layout != CSINN_LAYOUT_NCHW) {
        return CSINN_OPT_UNSUPPORTED;
    }
    int32_t input_h = input->dim[2];
    int32_t input_w = input->dim[3];

    int32_t kernel_h = params->filter_height;
    int32_t kernel_w = params->filter_width;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;
    int32_t pad_top = params->pad_top;
    int32_t pad_down = params->pad_down;
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        if (input_h == kernel_h && input_w == kernel_w) {
            return CSINN_OPT_ASM;
        }

        if (stride_h == 2 && stride_w == 2) {
            if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_ASM;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_ASM;
                } else {
                    return CSINN_OPT_C_REFERENCE;
                }
            } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_ASM;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_ASM;
                } else {
                    return CSINN_OPT_C_REFERENCE;
                }
            } else {
                return CSINN_OPT_C_REFERENCE;
            }
        } else if (stride_h == 1 && stride_w == 1 && kernel_h == 3 && kernel_w == 3 &&
                   pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        if (input_h == kernel_h && input_w == kernel_w) {
            return CSINN_OPT_ASM;
        }

        if (stride_h == 2 && stride_w == 2) {
            if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_ASM;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_ASM;
                } else {
                    return CSINN_OPT_C_REFERENCE;
                }
            } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_ASM;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_ASM;
                } else {
                    return CSINN_OPT_C_REFERENCE;
                }
            } else {
                return CSINN_OPT_C_REFERENCE;
            }
        } else if (stride_h == 1 && stride_w == 1 && kernel_h == 3 && kernel_w == 3 &&
                   pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else {
        return CSINN_OPT_C_REFERENCE;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_avgpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params)
{
    /* only support layout: NCHW */
    if (params->base.layout != CSINN_LAYOUT_NCHW) {
        return CSINN_OPT_UNSUPPORTED;
    }
    int32_t input_h = input->dim[2];
    int32_t input_w = input->dim[3];

    int32_t kernel_h = params->filter_height;
    int32_t kernel_w = params->filter_width;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;
    int32_t pad_top = params->pad_top;
    int32_t pad_down = params->pad_down;
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        if (input_h == kernel_h && input_w == kernel_w) {
            return CSINN_OPT_ASM;
        }

        if (stride_h == 2 && stride_w == 2) {
            if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_ASM;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_ASM;
                } else {
                    return CSINN_OPT_C_REFERENCE;
                }
            } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_ASM;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_ASM;
                } else {
                    return CSINN_OPT_C_REFERENCE;
                }
            } else {
                return CSINN_OPT_C_REFERENCE;
            }
        } else if (stride_h == 1 && stride_w == 1 && kernel_h == 3 && kernel_w == 3 &&
                   pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        if (input_h == kernel_h && input_w == kernel_w) {
            return CSINN_OPT_ASM;
        }

        if (stride_h == 2 && stride_w == 2) {
            if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_ASM;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_ASM;
                } else {
                    return CSINN_OPT_C_REFERENCE;
                }
            } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_ASM;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_ASM;
                } else {
                    return CSINN_OPT_C_REFERENCE;
                }
            } else {
                return CSINN_OPT_C_REFERENCE;
            }
        } else if (stride_h == 1 && stride_w == 1 && kernel_h == 3 && kernel_w == 3 &&
                   pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else {
        return CSINN_OPT_C_REFERENCE;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_div_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    if (input0->dtype == CSINN_DTYPE_FLOAT16) {
        if (input1->is_const) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else if (input0->dtype == CSINN_DTYPE_FLOAT32) {
        if (input1->is_const) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_abs_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params)
{
    return common_all_support(input, &(params->base));
}

static int c906_add_tail_coincide(struct csinn_tensor *input0, struct csinn_tensor *input1)
{
    int flag = 1;
    int i = 0, j = 0;
    for (i = input1->dim_count - 1, j = input0->dim_count - 1; i >= 0; i--, j--) {
        if (input0->dim[j] != input1->dim[i]) {
            flag = 0;
            break;
        }
    }
    flag = 1;
    for (; i >= 0; i--) {
        if (input1->dim[i] != 1) {
            flag = 0;
            break;
        }
    }
    return flag;
}

int shl_c906_add_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int in_size0 = csinn_tensor_size(input0);
    int in_size1 = csinn_tensor_size(input1);
    if (input0->dtype == CSINN_DTYPE_FLOAT16) {
        if ((input1->dim[2] == 1) && (input1->dim[3] == 1) && (input1->dim[1] == input0->dim[1])) {
            return CSINN_OPT_ASM;
        }
        if (in_size1 == 1) {
            return CSINN_OPT_ASM;
        } else if (in_size0 == in_size1) {
            return CSINN_OPT_ASM;
        } else if (c906_add_tail_coincide(input0, input1)) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else if (input0->dtype == CSINN_DTYPE_FLOAT32) {
        if ((input1->dim[2] == 1) && (input1->dim[3] == 1) && (input1->dim[1] == input0->dim[1])) {
            return CSINN_OPT_ASM;
        }
        if (in_size1 == 1) {
            return CSINN_OPT_ASM;
        } else if (in_size0 == in_size1) {
            return CSINN_OPT_ASM;
        } else if (c906_add_tail_coincide(input0, input1)) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_clip_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_clip_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c906_concat_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_clip_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c906_global_avgpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c906_global_maxpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c906_leaky_relu_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_relu_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c906_lrn_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_lrn_params *params)
{
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        return CSINN_OPT_C_REFERENCE;
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_matmul_cap(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    if (mat0->dtype == CSINN_DTYPE_FLOAT16) {
        return CSINN_OPT_ASM;
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_minimum_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params)
{
    return common_all_support(input0, &(params->base));
}

int shl_c906_mul_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int in_size0 = csinn_tensor_size(input0);
    int in_size1 = csinn_tensor_size(input1);
    if (input0->dtype == CSINN_DTYPE_FLOAT16) {
        if ((input1->dim[2] == 1) && (input1->dim[3] == 1) && (input1->dim[1] == input0->dim[1])) {
            return CSINN_OPT_ASM;
        }
        if (in_size1 == 1) {
            return CSINN_OPT_ASM;
        } else if (in_size0 == in_size1) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else if (input0->dtype == CSINN_DTYPE_FLOAT32) {
        if ((input1->dim[2] == 1) && (input1->dim[3] == 1) && (input1->dim[1] == input0->dim[1])) {
            return CSINN_OPT_ASM;
        }
        if (in_size1 == 1) {
            return CSINN_OPT_ASM;
        } else if (in_size0 == in_size1) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_prelu_cap(struct csinn_tensor *input, struct csinn_tensor *alpha,
                       struct csinn_tensor *output, struct csinn_prelu_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c906_relu_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c906_relu1_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c906_relu6_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c906_split_cap(struct csinn_tensor *input, struct csinn_tensor **output,
                       struct csinn_split_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_c906_sub_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int in_size0 = csinn_tensor_size(input0);
    int in_size1 = csinn_tensor_size(input1);
    if (input0->dtype == CSINN_DTYPE_FLOAT16) {
        if ((input1->dim[2] == 1) && (input1->dim[3] == 1) && (input1->dim[1] == input0->dim[1])) {
            return CSINN_OPT_ASM;
        }
        if (in_size1 == 1) {
            return CSINN_OPT_ASM;
        } else if (in_size0 == in_size1) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else if (input0->dtype == CSINN_DTYPE_FLOAT32) {
        if ((input1->dim[2] == 1) && (input1->dim[3] == 1) && (input1->dim[1] == input0->dim[1])) {
            return CSINN_OPT_ASM;
        }
        if (in_size1 == 1) {
            return CSINN_OPT_ASM;
        } else if (in_size0 == in_size1) {
            return CSINN_OPT_ASM;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_reshape_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params)
{
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        return CSINN_OPT_ASM;
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_c906_sum_stride_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params)
{
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        return CSINN_OPT_ASM;
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_UNSUPPORTED;
}
