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

static int common_all_support(struct csinn_tensor *input, struct csinn_params_base *base)
{
    if ((input->dtype != CSINN_DTYPE_FLOAT16) && (input->dtype != CSINN_DTYPE_FLOAT32) &&
        (input->dtype != CSINN_DTYPE_INT8)) {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_INTRINSIC;
}

static int float_all_support(struct csinn_tensor *input, struct csinn_params_base *base)
{
    if ((input->dtype != CSINN_DTYPE_FLOAT16) && (input->dtype != CSINN_DTYPE_FLOAT32)) {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_INTRINSIC;
}

int shl_rvv_conv2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv2d_params *params)
{
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        return CSINN_OPT_INTRINSIC;
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        return CSINN_OPT_INTRINSIC;
    } else if (input->dtype == CSINN_DTYPE_INT8) {
        return CSINN_OPT_INTRINSIC;
    } else if (input->dtype == CSINN_DTYPE_INT4) {
        if (input->layout == CSINN_LAYOUT_NHWC) {
            return CSINN_OPT_INTRINSIC;
        }
    }
    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_depthwise_conv2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params)
{
    int32_t in_c = input->dim[1];
    int32_t out_c = output->dim[1];
    int32_t kernel_h = kernel->dim[2];
    int32_t kernel_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        const int packn = csrr_vlenb() / sizeof(__fp16);
        if (in_c % packn == 0 && out_c % packn == 0) {
            return CSINN_OPT_INTRINSIC;
        }

        if (in_c % packn != 0 && out_c % packn != 0) {
            if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
                return CSINN_OPT_INTRINSIC;
            } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
                return CSINN_OPT_INTRINSIC;
            } else {
                return CSINN_OPT_C_REFERENCE;
            }
        }
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        const int packn = csrr_vlenb() / sizeof(float);
        if (in_c % packn == 0 && out_c % packn == 0) {
            if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
                return CSINN_OPT_INTRINSIC;
            }
        }

        if (in_c % packn != 0 && out_c % packn != 0) {
            if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
                return CSINN_OPT_INTRINSIC;
            } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
                return CSINN_OPT_INTRINSIC;
            } else {
                return CSINN_OPT_C_REFERENCE;
            }
        }
    } else if (input->dtype == CSINN_DTYPE_INT8) {
        const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
        if (in_c % packn == 0 && out_c % packn == 0) {
            return CSINN_OPT_INTRINSIC;
        }

        if (in_c % packn != 0 && out_c % packn != 0) {
            if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
                return CSINN_OPT_INTRINSIC;
            } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
                return CSINN_OPT_INTRINSIC;
            } else {
                return CSINN_OPT_C_REFERENCE;
            }
        }
    } else if (input->dtype == CSINN_DTYPE_INT4) {
        if (input->layout == CSINN_LAYOUT_NHWC) {
            if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
                return CSINN_OPT_INTRINSIC;
            } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
                return CSINN_OPT_INTRINSIC;
            }
        }
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_conv1d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv1d_params *params)
{
    int32_t kernel_w = kernel->dim[2];
    int32_t stride_w = params->stride_width;
    int32_t dilation_w = params->dilation_width;
    int32_t group = params->group;
    if (input->dtype == CSINN_DTYPE_FLOAT32) {
        if (group == 1) {
            return CSINN_OPT_INTRINSIC;
        }
        // dwconv1d
        else if (group == input->dim[1] && kernel->dim[1] == 1) {
            if (bias->data != NULL && bias->dim_count != 0) {
                return CSINN_OPT_INTRINSIC;
            } else {
                return CSINN_OPT_C_REFERENCE;
            }
        }
        // group conv1d
        else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
        if (group == 1) {
            return CSINN_OPT_INTRINSIC;
        }
        // dwconv1d
        else if (group == input->dim[1] && kernel->dim[1] == 1) {
            if (bias->data != NULL && bias->dim_count != 0) {
                return CSINN_OPT_INTRINSIC;
            } else {
                return CSINN_OPT_C_REFERENCE;
            }
        }
        // group conv1d
        else {
            return CSINN_OPT_C_REFERENCE;
        }
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_deconv2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv2d_params *params)
{
    if (input->dtype == CSINN_DTYPE_FLOAT32) {
        return CSINN_OPT_INTRINSIC;
    } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
        return CSINN_OPT_INTRINSIC;
    } else {
        return CSINN_OPT_UNSUPPORTED;
    }
}

int shl_rvv_fullyconnected_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weights, struct csinn_tensor *bias,
                               struct csinn_fc_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_maxpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_pool_params *params)
{
    int32_t in_c = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t kernel_h = params->filter_height;
    int32_t kernel_w = params->filter_width;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;
    int32_t pad_top = params->pad_top;
    int32_t pad_down = params->pad_down;
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        const int packn = csrr_vlenb() / sizeof(__fp16);
        if (in_h == kernel_h && in_w == kernel_w) {
            return CSINN_OPT_INTRINSIC;
        }

        if (stride_h == 2 && stride_w == 2) {
            if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        } else if (stride_h == 1 && stride_w == 1) {
            if (kernel_h == 3 && kernel_w == 3) {
                if (pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        }

        if (in_c % packn == 0) {
            return CSINN_OPT_INTRINSIC;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }

    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        const int packn = csrr_vlenb() / sizeof(float);
        if (in_h == kernel_h && in_w == kernel_w) {
            return CSINN_OPT_INTRINSIC;
        }

        if (stride_h == 2 && stride_w == 2) {
            if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        } else if (stride_h == 1 && stride_w == 1) {
            if (kernel_h == 3 && kernel_w == 3) {
                if (pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        }

        if (in_c % packn == 0) {
            return CSINN_OPT_INTRINSIC;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else if (input->dtype == CSINN_DTYPE_INT8) {
        const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
        if (in_h == kernel_h && in_w == kernel_w) {
            return CSINN_OPT_INTRINSIC;
        }

        if (stride_h == 2 && stride_w == 2) {
            if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        } else if (stride_h == 1 && stride_w == 1) {
            if (kernel_h == 3 && kernel_w == 3) {
                if (pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        }

        if (in_c % packn == 0) {
            return CSINN_OPT_INTRINSIC;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_avgpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_pool_params *params)
{
    int32_t in_c = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t kernel_h = params->filter_height;
    int32_t kernel_w = params->filter_width;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;
    int32_t pad_top = params->pad_top;
    int32_t pad_down = params->pad_down;
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        const int packn = csrr_vlenb() / sizeof(__fp16);
        if (in_h == kernel_h && in_w == kernel_w) {
            return CSINN_OPT_INTRINSIC;
        }

        if (stride_h == 2 && stride_w == 2) {
            if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        } else if (stride_h == 1 && stride_w == 1) {
            if (kernel_h == 3 && kernel_w == 3) {
                if (pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        }

        if (in_c % packn == 0) {
            return CSINN_OPT_INTRINSIC;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }

    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        const int packn = csrr_vlenb() / sizeof(float);
        if (in_h == kernel_h && in_w == kernel_w) {
            return CSINN_OPT_INTRINSIC;
        }

        if (stride_h == 2 && stride_w == 2) {
            if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        } else if (stride_h == 1 && stride_w == 1) {
            if (kernel_h == 3 && kernel_w == 3) {
                if (pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        }

        if (in_c % packn == 0) {
            return CSINN_OPT_INTRINSIC;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    } else if (input->dtype == CSINN_DTYPE_INT8) {
        const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
        if (in_h == kernel_h && in_w == kernel_w) {
            return CSINN_OPT_INTRINSIC;
        }

        if (stride_h == 2 && stride_w == 2) {
            if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
                if (pad_left == 0 && pad_top == 0) {
                    return CSINN_OPT_INTRINSIC;
                } else if (pad_left == 1 && pad_top == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        } else if (stride_h == 1 && stride_w == 1) {
            if (kernel_h == 3 && kernel_w == 3) {
                if (pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
                    return CSINN_OPT_INTRINSIC;
                }
            }
        }

        if (in_c % packn == 0) {
            return CSINN_OPT_INTRINSIC;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_add_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params)
{
    if (input0->dtype == CSINN_DTYPE_FLOAT16) {
        return CSINN_OPT_INTRINSIC;
    } else if (input0->dtype == CSINN_DTYPE_FLOAT32) {
        return CSINN_OPT_INTRINSIC;
    } else if (input0->dtype == CSINN_DTYPE_INT8) {
        return CSINN_OPT_INTRINSIC;
    }
    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_sub_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params)
{
    if (input0->dtype == CSINN_DTYPE_FLOAT16) {
        return CSINN_OPT_INTRINSIC;
    } else if (input0->dtype == CSINN_DTYPE_FLOAT32) {
        return CSINN_OPT_INTRINSIC;
    } else if (input0->dtype == CSINN_DTYPE_INT8) {
        return CSINN_OPT_INTRINSIC;
    }
    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_mul_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params)
{
    if (input0->dtype == CSINN_DTYPE_FLOAT16) {
        return CSINN_OPT_INTRINSIC;
    } else if (input0->dtype == CSINN_DTYPE_FLOAT32) {
        return CSINN_OPT_INTRINSIC;
    } else if (input0->dtype == CSINN_DTYPE_INT8) {
        return CSINN_OPT_INTRINSIC;
    }
    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_div_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params)
{
    if (input0->dtype == CSINN_DTYPE_FLOAT16) {
        return CSINN_OPT_INTRINSIC;
    } else if (input0->dtype == CSINN_DTYPE_FLOAT32) {
        return CSINN_OPT_INTRINSIC;
    } else if (input0->dtype == CSINN_DTYPE_INT8) {
        return CSINN_OPT_INTRINSIC;
    }
    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_concat_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_clip_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_leaky_relu_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_relu_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_relu_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_relu6_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_global_avgpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_global_maxpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_reshape_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_reshape_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_sigmoid_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_sigmoid_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_softmax_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_softmax_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_reduce_sum_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params)
{
    if (input->dtype == CSINN_DTYPE_INT8) {
        return CSINN_OPT_INTRINSIC;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_prelu_cap(struct csinn_tensor *input, struct csinn_tensor *alpha,
                      struct csinn_tensor *output, struct csinn_prelu_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_layer_norm_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *gamma, struct csinn_tensor *beta,
                           struct csinn_layer_norm_params *params)
{
    if (params->center == false || params->scale == false) {
        return CSINN_OPT_UNSUPPORTED;
    }
    return common_all_support(input, &(params->base));
}

int shl_rvv_clip_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_clip_params *params)
{
    return common_all_support(input, &(params->base));
}

int shl_rvv_transpose_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_transpose_params *params)
{
    int tail = shl_rvv_transpose_get_tail(params->permute, params->permute_num);
    if (input->dtype == CSINN_DTYPE_FLOAT32) {
        if (params->permute_num == 4 && params->permute[0] == 0 && params->permute[1] == 1 &&
            params->permute[2] == 2 && params->permute[3] == 3) {
            return CSINN_OPT_INTRINSIC;
        } else if (params->permute_num == 4 && params->permute[0] == 0 && params->permute[1] == 2 &&
                   params->permute[2] == 3 && params->permute[3] == 1) {
            return CSINN_OPT_INTRINSIC;
        } else if (params->permute_num == 3 && params->permute[0] == 0 && params->permute[1] == 2 &&
                   params->permute[2] == 1) {
            return CSINN_OPT_INTRINSIC;
        } else if (tail > 0) {
            return CSINN_OPT_INTRINSIC;
        }
        return CSINN_OPT_C_REFERENCE;
    } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
        if (params->permute_num == 4 && params->permute[0] == 0 && params->permute[1] == 1 &&
            params->permute[2] == 2 && params->permute[3] == 3) {
            return CSINN_OPT_INTRINSIC;
        } else if (params->permute_num == 4 && params->permute[0] == 0 && params->permute[1] == 2 &&
                   params->permute[2] == 3 && params->permute[3] == 1) {
            return CSINN_OPT_INTRINSIC;
        } else if (params->permute_num == 3 && params->permute[0] == 0 && params->permute[1] == 2 &&
                   params->permute[2] == 1) {
            return CSINN_OPT_INTRINSIC;
        } else if (tail > 0) {
            return CSINN_OPT_INTRINSIC;
        }
        return CSINN_OPT_C_REFERENCE;
    } else if (input->dtype == CSINN_DTYPE_INT8) {
        if (params->permute_num == 4 && params->permute[0] == 0 && params->permute[1] == 1 &&
            params->permute[2] == 2 && params->permute[3] == 3) {
            return CSINN_OPT_INTRINSIC;
        } else if (params->permute_num == 4 && params->permute[0] == 0 && params->permute[1] == 2 &&
                   params->permute[2] == 3 && params->permute[3] == 1) {
            return CSINN_OPT_INTRINSIC;
        } else if (params->permute_num == 3 && params->permute[0] == 0 && params->permute[1] == 2 &&
                   params->permute[2] == 1) {
            return CSINN_OPT_INTRINSIC;
        } else if (tail > 0) {
            return CSINN_OPT_INTRINSIC;
        }
        return CSINN_OPT_C_REFERENCE;
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_matmul_cap(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                       struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    int batches_a = 1;
    int batches_b = 1;

    /* compute the outer size */
    for (int i = 0; i < mat0->dim_count - 2; i++) {
        batches_a *= mat0->dim[i];
    }
    for (int i = 0; i < mat1->dim_count - 2; i++) {
        batches_b *= mat1->dim[i];
    }

    if (mat0->dtype == CSINN_DTYPE_FLOAT32 && mat1->dtype == CSINN_DTYPE_FLOAT32 ||
        mat0->dtype == CSINN_DTYPE_FLOAT16 &&
            (mat1->dtype == CSINN_DTYPE_FLOAT16 || mat1->dtype == CSINN_DTYPE_INT8)) {
        if (!params->trans_a && !params->trans_b) {
            if (batches_a == batches_b) {
                return CSINN_OPT_INTRINSIC;
            } else if (batches_a > 1 && batches_b == 1) {
                return CSINN_OPT_INTRINSIC;
            }
        }
    }

    if (mat0->dtype == CSINN_DTYPE_INT8) {
        if (batches_a == batches_b) {
            return CSINN_OPT_INTRINSIC;
        } else if (batches_a > 1 && batches_b == 1) {
            if (!params->trans_a && !params->trans_b) {
                return CSINN_OPT_INTRINSIC;
            }
        }
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_gather_cap(struct csinn_tensor *input, struct csinn_tensor *indices,
                       struct csinn_tensor *output, struct csinn_gather_params *params)
{
    if (input->dtype == CSINN_DTYPE_FLOAT32) {
        if (indices->dtype == CSINN_DTYPE_INT64 && output->dtype == CSINN_DTYPE_FLOAT32) {
            return CSINN_OPT_INTRINSIC;
        }
    } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
        if (indices->dtype == CSINN_DTYPE_INT64 && output->dtype == CSINN_DTYPE_FLOAT16) {
            return CSINN_OPT_INTRINSIC;
        }
    } else if (input->dtype == CSINN_DTYPE_INT8) {
        if (indices->dtype == CSINN_DTYPE_INT64 && output->dtype == CSINN_DTYPE_FLOAT16) {
            return CSINN_OPT_INTRINSIC;
        } else if (indices->dtype == CSINN_DTYPE_INT64 && output->dtype == CSINN_DTYPE_INT8) {
            return CSINN_OPT_INTRINSIC;
        } else {
            return CSINN_OPT_C_REFERENCE;
        }
    }

    return CSINN_OPT_UNSUPPORTED;
}

int shl_rvv_erf_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_clip_params *params)
{
    return common_all_support(input, &(params->base));
}
