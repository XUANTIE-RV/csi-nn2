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

#ifdef SHL_EXPORT_MODEL

#include "export_json_wrapper.h"
extern "C" {
#include "csi_nn.h"
#include "shl_gref.h"
}
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

#include "json/json.hpp"
using json = nlohmann::ordered_json;

NLOHMANN_JSON_SERIALIZE_ENUM(csinn_mem_type_enum,
                             {
                                 {CSINN_MEM_TYPE_CPU_NOT_ALIGNED, "CSINN_MEM_TYPE_CPU_NOT_ALIGNED"},
                                 {CSINN_MEM_TYPE_CPU_ALIGNED, "CSINN_MEM_TYPE_CPU_ALIGNED"},
                                 {CSINN_MEM_TYPE_DMABUF, "CSINN_MEM_TYPE_DMABUF"},
                                 {CSINN_MEM_TYPE_ASP42, "CSINN_MEM_TYPE_ASP42"},
                                 {CSINN_MEM_TYPE_ASP41, "CSINN_MEM_TYPE_ASP41"},
                                 {CSINN_MEM_TYPE_CPU_ACC, "CSINN_MEM_TYPE_CPU_ACC"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(
    csinn_dtype_enum,
    {
        {CSINN_DTYPE_BOOL, "CSINN_DTYPE_BOOL"},         /**< Boolean */
        {CSINN_DTYPE_INT4, "CSINN_DTYPE_INT4"},         /**< Signed 4 bit fixed-point */
        {CSINN_DTYPE_UINT8, "CSINN_DTYPE_UINT8"},       /**< Unsigned 8 bit fixed-point */
        {CSINN_DTYPE_INT8, "CSINN_DTYPE_INT8"},         /**< Signed 8 bit fixed-point */
        {CSINN_DTYPE_UINT16, "CSINN_DTYPE_UINT16"},     /**< Unsigned 16 bit fixed-point */
        {CSINN_DTYPE_INT16, "CSINN_DTYPE_INT16"},       /**< Signed 16 bit fixed-point */
        {CSINN_DTYPE_UINT32, "CSINN_DTYPE_UINT32"},     /**< Unsigned 32 bit fixed-point */
        {CSINN_DTYPE_INT32, "CSINN_DTYPE_INT32"},       /**< Signed 32 bit fixed-point */
        {CSINN_DTYPE_FLOAT16, "CSINN_DTYPE_FLOAT16"},   /**< Half-precision floating-point */
        {CSINN_DTYPE_BFLOAT16, "CSINN_DTYPE_BFLOAT16"}, /**< Brain floating-point */
        {CSINN_DTYPE_FLOAT32, "CSINN_DTYPE_FLOAT32"},   /**< Single-precision floating-point */
        {CSINN_DTYPE_FLOAT64, "CSINN_DTYPE_FLOAT64"},   /**< Double-precision floating-point */
        {CSINN_DTYPE_INT64, "CSINN_DTYPE_INT64"},       /**< Signed 64 bit fixed-point */
    })

NLOHMANN_JSON_SERIALIZE_ENUM(
    csinn_quant_enum,
    {
        {CSINN_QUANT_UNSET, "CSINN_QUANT_UNSET"}, /**< The quantization type is not set */
        {CSINN_QUANT_INT4_SYM,
         "CSINN_QUANT_INT4_SYM"}, /**< Symmetric signed 4-bit fixed-point quantization */
        {CSINN_QUANT_UINT8_ASYM,
         "CSINN_QUANT_UINT8_ASYM"}, /**< Asymmetric unsigned 8-bit fixed-point quantization */
        {CSINN_QUANT_UINT8_SYM,
         "CSINN_QUANT_UINT8_SYM"}, /**< Symmetric unsigned 8-bit fixed-point quantization */
        {CSINN_QUANT_INT8_ASYM,
         "CSINN_QUANT_INT8_ASYM"}, /**< Asymmetric signed 8-bit fixed-point quantization */
        {CSINN_QUANT_INT8_SYM,
         "CSINN_QUANT_INT8_SYM"}, /**< Symmetric signed 8-bit fixed-point quantization */
        {CSINN_QUANT_INT16_SYM,
         "CSINN_QUANT_INT16_SYM"}, /**< Symmetric signed 16-bit fixed-point quantization */
        {CSINN_QUANT_FLOAT16, "CSINN_QUANT_FLOAT16"},   /**< 16-bit floating-point quantization */
        {CSINN_QUANT_BFLOAT16, "CSINN_QUANT_BFLOAT16"}, /**< bf16 floating-point quantization */
        {CSINN_QUANT_FLOAT32, "CSINN_QUANT_FLOAT32"},   /**< 32-bit floating-point not quantized */
        {CSINN_QUANT_INT4_ASYM_W_SYM,
         "CSINN_QUANT_INT4_ASYM_W_SYM"}, /**< Signed 4-bit Asymmetric activation and Symmetric
                                            weight */
        {CSINN_QUANT_INT8_ASYM_W_SYM,
         "CSINN_QUANT_INT8_ASYM_W_SYM"}, /**< Signed 8-bit Asymmetric activation and Symmetric
                                            weight */
        {CSINN_QUANT_FLOAT16_W_INT8,
         "CSINN_QUANT_FLOAT16_W_INT8"}, /**< 16-bit floating-point and 8-bit symmetric weight */
    })

NLOHMANN_JSON_SERIALIZE_ENUM(
    csinn_api_enum, {
                        {CSINN_REF, "CSINN_REF"},           /**< Reference c */
                        {CSINN_GREF, "CSINN_GREF"},         /**< reference graph */
                        {CSINN_C860, "CSINN_C860"},         /**< C860 CPU platform */
                        {CSINN_C906, "CSINN_C906"},         /**< C906 CPU platform */
                        {CSINN_C920, "CSINN_C920"},         /**< C920 CPU platform */
                        {CSINN_ANOLE, "CSINN_ANOLE"},       /**< anole NPU platform */
                        {CSINN_CH8601, "CSINN_CH8601"},     /**< ch8601 NPU platform */
                        {CSINN_TH1520, "CSINN_TH1520"},     /**< th1520 NPU platform */
                        {CSINN_DP1K, "CSINN_DP1K"},         /**< dp1000 NPU platform */
                        {CSINN_I805, "CSINN_I805"},         /**< I805 CPU platform */
                        {CSINN_E804, "CSINN_E804"},         /**< E804 CPU platform */
                        {CSINN_REF_I805, "CSINN_REF_I805"}, /**< I805 CPU platform */
                        {CSINN_C908, "CSINN_C908"},         /**< C908 CPU platform */
                        {CSINN_TVMGEN, "CSINN_TVMGEN"},     /**< TVM generate platform */
                        {CSINN_ASP, "CSINN_ASP"},           /**< ASP platform */
                        {CSINN_RVV, "CSINN_RVV"},   /**< RISC-V V extension general platform */
                        {CSINN_RVM, "CSINN_RVM"},   /**< RISC-V Matrix extension general platform */
                        {CSINN_E907, "CSINN_E907"}, /**< E907 CPU platform */
                    })

NLOHMANN_JSON_SERIALIZE_ENUM(
    csinn_layout_enum,
    {
        {CSINN_LAYOUT_NULL, "CSINN_LAYOUT_NULL"}, /**< Not set */
        // NCHW
        // ACTIVITION
        {CSINN_LAYOUT_N, "CSINN_LAYOUT_N"},         /**< NCHW input and output, 1 dimension */
        {CSINN_LAYOUT_NC, "CSINN_LAYOUT_NC"},       /**< NCHW input and output, 2 dimensions */
        {CSINN_LAYOUT_NCW, "CSINN_LAYOUT_NCW"},     /**< NCHW input and output, 3 dimensions */
        {CSINN_LAYOUT_NCHW, "CSINN_LAYOUT_NCHW"},   /**< NCHW input and output, 4 dimensions */
        {CSINN_LAYOUT_NCDHW, "CSINN_LAYOUT_NCDHW"}, /**< NCHW input and output, 5 dimensions */
        // WEIGHT
        {CSINN_LAYOUT_O, "CSINN_LAYOUT_O"},           /**< NCHW constant, 1 dimension */
        {CSINN_LAYOUT_OI, "CSINN_LAYOUT_OI"},         /**< NCHW constant, 2 dimensions */
        {CSINN_LAYOUT_O16I16, "CSINN_LAYOUT_O16I16"}, /**< 16 bytes in parallel for ASP platform */
        {CSINN_LAYOUT_O32I32, "CSINN_LAYOUT_O32I32"}, /**< 32 bytes in parallel for ASP platform */
        {CSINN_LAYOUT_OIW, "CSINN_LAYOUT_OIW"},       /**< NCHW constant, 3 dimension */
        {CSINN_LAYOUT_OIHW, "CSINN_LAYOUT_OIHW"},     /**< NCHW constant, 4 dimension */
        {CSINN_LAYOUT_IOHW, "CSINN_LAYOUT_IOHW"},     /**< NCHW constant, 4 dimension */
        {CSINN_LAYOUT_OIDHW, "CSINN_LAYOUT_OIDHW"},   /**< NCHW constant, 5 dimension */
        {CSINN_LAYOUT_O1HW, "CSINN_LAYOUT_O1HW"}, /**< NCHW constant, depthwise convolution only */

        // NHWC
        // ACTIVITION
        {CSINN_LAYOUT_NWC, "CSINN_LAYOUT_NWC"},     /**< NHWC input and output, 3 dimensions */
        {CSINN_LAYOUT_NHWC, "CSINN_LAYOUT_NHWC"},   /**< NHWC input and output, 4 dimensions */
        {CSINN_LAYOUT_NDHWC, "CSINN_LAYOUT_NDHWC"}, /**< NHWC input and output, 5 dimensions */
        // WEIGHT
        {CSINN_LAYOUT_OWI, "CSINN_LAYOUT_OWI"},   /**< NHWC constant, 3 dimensions */
        {CSINN_LAYOUT_OHWI, "CSINN_LAYOUT_OHWI"}, /**< NHWC constant, 4 dimensions */
        {CSINN_LAYOUT_O16HWI16,
         "CSINN_LAYOUT_O16HWI16"}, /**< 16 bytes in parallel for ASP platform */
        {CSINN_LAYOUT_O32HWI32,
         "CSINN_LAYOUT_O32HWI32"},                  /**< 32 bytes in parallel for ASP platform */
        {CSINN_LAYOUT_ODHWI, "CSINN_LAYOUT_ODHWI"}, /**< NHWC constant, 5 dimensions */
        {CSINN_LAYOUT_1HWO, "CSINN_LAYOUT_1HWO"}, /**< NHWC constant, depthwise convolution only */
        {CSINN_LAYOUT_1HW16O16,
         "CSINN_LAYOUT_1HW16O16"}, /**< 16 bytes in parallel for ASP platform */
        {CSINN_LAYOUT_1HW32O32,
         "CSINN_LAYOUT_1HW32O32"}, /**< 32 bytes in parallel for ASP platform */

        // NC1HWC0
        // ACTIVITION
        // RVV optimization format: c0=4/8/8 for fp32/fp16/int8 when vlen=128
        {CSINN_LAYOUT_NC1C0, "CSINN_LAYOUT_NC1C0"},   /**< NC1HWC0 input and output, 2 dimension */
        {CSINN_LAYOUT_NC1WC0, "CSINN_LAYOUT_NC1WC0"}, /**< NC1HWC0 input and output, 3 dimension */
        {CSINN_LAYOUT_NC1HWC0,
         "CSINN_LAYOUT_NC1HWC0"}, /**< NC1HWC0 input and output, 4 dimension */
        {CSINN_LAYOUT_NC1DHWC0,
         "CSINN_LAYOUT_NC1DHWC0"}, /**< NC1HWC0 input and output, 5 dimension */

        // for 6D shape
        {CSINN_LAYOUT_NLCDHW, "CSINN_LAYOUT_NLCDHW"}, /**< NCHW input and output, 6 dimensions */
    })

NLOHMANN_JSON_SERIALIZE_ENUM(
    csinn_lrn_enum,
    {
        {CSINN_LRN_ACROSS_CHANNELS,
         "CSINN_LRN_ACROSS_CHANNELS"}, /**< local response normalization across channels/channels */
        {CSINN_LRN_WITHIN_CHANNEL,
         "CSINN_LRN_WITHIN_CHANNEL"}, /**< local response normalization within the same channel */
    })

static json shl_export_json_tensor(struct csinn_tensor *tensor)
{
    nlohmann::ordered_map<std::string, json> tensor_j;
    tensor_j["name"] = tensor->name;
    tensor_j["dtype"] = tensor->dtype;
    tensor_j["mtype"] = tensor->mtype;
    std::vector<int32_t> dim(tensor->dim, tensor->dim + tensor->dim_count);
    tensor_j["dim"] = dim;
    tensor_j["is_const"] = tensor->is_const;
    tensor_j["layout"] = (enum csinn_layout_enum)tensor->layout;

    if (tensor->dtype != CSINN_DTYPE_FLOAT32 && tensor->dtype != CSINN_DTYPE_FLOAT64 &&
        tensor->dtype != CSINN_DTYPE_INT64) {
        tensor_j["quant_channel"] = tensor->quant_channel;
        tensor_j["quant_info"] = {};
        for (int i = 0; tensor->quant_channel; i++) {
            json quant_info;
            quant_info["scale"] = tensor->qinfo[i].scale;
            quant_info["zero_point"] = tensor->qinfo[i].zero_point;
            quant_info["multiplier"] = tensor->qinfo[i].multiplier;
            quant_info["shift"] = tensor->qinfo[i].shift;
            quant_info["min"] = tensor->qinfo[i].min;
            quant_info["max"] = tensor->qinfo[i].max;

            tensor_j["quant_info"].push_back(quant_info);
        }
    }

    return tensor_j;
}

static void shl_export_json_input_tensor(std::vector<struct csinn_tensor *> in_tensor, json &jobj)
{
    jobj["inputs"] = {};
    for (auto t : in_tensor) {
        jobj["inputs"].push_back(shl_export_json_tensor(t));
    }
}

static void shl_export_json_output_tensor(std::vector<struct csinn_tensor *> out_tensor, json &jobj)
{
    jobj["outputs"] = {};
    for (auto t : out_tensor) {
        jobj["outputs"].push_back(shl_export_json_tensor(t));
    }
}

static void shl_export_json_params_base(struct csinn_params_base base, json &jobj)
{
    jobj["name"] = base.name;
    jobj["quant_type"] = base.quant_type;
    jobj["api"] = (enum csinn_api_enum)base.api;
}

static int shl_export_json_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params, std::string op_type,
                                  json &jobj)
{
    json conv_data;
    conv_data["op_type"] = op_type;
    shl_export_json_params_base(params->base, conv_data);

    json attrs;
    attrs["group"] = params->group;
    attrs["stride_height"] = params->stride_height;
    attrs["stride_width"] = params->stride_width;
    attrs["pad_top"] = params->pad_top;
    attrs["pad_left"] = params->pad_left;
    attrs["pad_down"] = params->pad_down;
    attrs["pad_right"] = params->pad_right;
    attrs["dilation_height"] = params->dilation_height;
    attrs["dilation_width"] = params->dilation_width;
    attrs["out_pad_height"] = params->out_pad_height;
    attrs["out_pad_width"] = params->out_pad_width;

    conv_data["attrs"] = attrs;

    // insert input info
    std::vector<struct csinn_tensor *> in_tensors = {input, kernel, bias};
    shl_export_json_input_tensor(in_tensors, conv_data);

    // insert output info
    std::vector<struct csinn_tensor *> out_tensors = {output};
    shl_export_json_output_tensor(out_tensors, conv_data);

    jobj["layers"].push_back(conv_data);

    return CSINN_TRUE;
}

static int shl_export_json_siso(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_params_base *params, std::string op_type, json &jobj)
{
    json siso_data;
    siso_data["op_type"] = op_type;
    shl_export_json_params_base(*params, siso_data);

    // generate input info
    std::vector<struct csinn_tensor *> in_tensors = {input};
    shl_export_json_input_tensor(in_tensors, siso_data);

    // generate output info
    std::vector<struct csinn_tensor *> out_tensors = {output};
    shl_export_json_output_tensor(out_tensors, siso_data);

    jobj["layers"].push_back(siso_data);

    return CSINN_TRUE;
}

static int shl_export_json_softmax(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_softmax_params *params, std::string op_type,
                                   json &jobj)
{
    json softmax_data;
    softmax_data["op_type"] = op_type;
    shl_export_json_params_base(params->base, softmax_data);

    // generate attrs
    json attrs;
    attrs["axis"] = params->axis;

    // generate input info
    shl_export_json_input_tensor({input}, softmax_data);

    // generate output_info
    shl_export_json_output_tensor({output}, softmax_data);

    jobj["layers"].push_back(softmax_data);

    return CSINN_TRUE;
}

static int shl_export_json_diso(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                struct csinn_tensor *output, struct csinn_diso_params *params,
                                std::string op_type, json &jobj)
{
    json diso_data;
    diso_data["op_type"] = op_type;
    shl_export_json_params_base(params->base, diso_data);

    // generate input info
    std::vector<struct csinn_tensor *> in_tensors = {input0, input1};
    shl_export_json_input_tensor(in_tensors, diso_data);

    // generate output info
    std::vector<struct csinn_tensor *> out_tensors = {output};
    shl_export_json_output_tensor(out_tensors, diso_data);

    jobj["layers"].push_back(diso_data);

    return CSINN_TRUE;
}

static int shl_export_json_pool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params, std::string op_type, json &jobj)
{
    json pool_data;
    pool_data["op_type"] = op_type;
    shl_export_json_params_base(params->base, pool_data);

    // generate attrs info
    json attrs;
    attrs["pool_type"] = params->pool_type;
    attrs["filter_height"] = params->filter_height;
    attrs["filter_width"] = params->filter_width;
    attrs["filter_depth"] = params->filter_depth;
    attrs["stride_height"] = params->stride_height;
    attrs["stride_width"] = params->stride_width;
    attrs["stride_depth"] = params->stride_depth;
    attrs["pad_top"] = params->pad_top;
    attrs["pad_left"] = params->pad_left;
    attrs["pad_down"] = params->pad_down;
    attrs["pad_right"] = params->pad_right;
    attrs["pad_front"] = params->pad_front;
    attrs["pad_back"] = params->pad_back;
    attrs["ceil_mode"] = params->ceil_mode;
    attrs["count_include_pad"] = params->count_include_pad;
    pool_data["attrs"] = attrs;

    // generate input info
    std::vector<struct csinn_tensor *> in_tensors = {input};
    shl_export_json_input_tensor(in_tensors, pool_data);

    // generate output info
    std::vector<struct csinn_tensor *> out_tensors = {output};
    shl_export_json_output_tensor(out_tensors, pool_data);

    jobj["layers"].push_back(pool_data);

    return CSINN_TRUE;
}

static int shl_export_json_reshape(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_reshape_params *params, std::string op_type,
                                   json &jobj)
{
    json reshape_data;
    reshape_data["op_type"] = op_type;
    shl_export_json_params_base(params->base, reshape_data);

    // generate attrs
    json attrs;
    std::vector<int32_t> shape(params->shape, params->shape + params->shape_num);
    attrs["reshape"] = shape;
    reshape_data["attrs"] = attrs;

    // generate input info
    shl_export_json_input_tensor({input}, reshape_data);

    // generate output_info
    shl_export_json_output_tensor({output}, reshape_data);

    jobj["layers"].push_back(reshape_data);

    return CSINN_TRUE;
}

static int shl_export_json_fullyconnected(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_fc_params *params, std::string op_type,
                                          json &jobj)
{
    json fcl_data;
    fcl_data["op_type"] = op_type;
    shl_export_json_params_base(params->base, fcl_data);

    json attrs;
    attrs["units"] = params->units;
    fcl_data["attrs"] = attrs;

    // insert input info
    std::vector<struct csinn_tensor *> in_tensors = {input, kernel, bias};
    shl_export_json_input_tensor(in_tensors, fcl_data);

    // insert output info
    std::vector<struct csinn_tensor *> out_tensors = {output};
    shl_export_json_output_tensor(out_tensors, fcl_data);

    jobj["layers"].push_back(fcl_data);

    return CSINN_TRUE;
}

static int shl_export_json_lrn(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_lrn_params *params, std::string op_type, json &jobj)
{
    json lrn_data;
    lrn_data["op_type"] = op_type;
    shl_export_json_params_base(params->base, lrn_data);

    // generate attrs
    json attrs;
    attrs["range"] = params->range;
    attrs["bias"] = params->bias;
    attrs["alpha"] = params->alpha;
    attrs["beta"] = params->beta;
    attrs["norm_region"] = params->norm_region;
    lrn_data["attrs"] = attrs;

    // generate input info
    shl_export_json_input_tensor({input}, lrn_data);

    // generate output_info
    shl_export_json_output_tensor({output}, lrn_data);

    jobj["layers"].push_back(lrn_data);

    return CSINN_TRUE;
}

static int shl_export_json_concat(struct csinn_tensor **input, struct csinn_tensor *output,
                                  struct csinn_concat_params *params, std::string op_type,
                                  json &jobj)
{
    json concat_data;
    concat_data["op_type"] = op_type;
    shl_export_json_params_base(params->base, concat_data);

    // generate attrs
    json attrs;
    attrs["inputs_count"] = params->inputs_count;
    attrs["axis"] = params->axis;
    concat_data["attrs"] = attrs;

    // generate input info
    shl_export_json_input_tensor(
        std::vector<struct csinn_tensor *>(input, input + params->inputs_count), concat_data);

    // generate output_info
    shl_export_json_output_tensor({output}, concat_data);

    jobj["layers"].push_back(concat_data);

    return CSINN_TRUE;
}

static int shl_export_json_prelu(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                 struct csinn_tensor *output, struct csinn_prelu_params *params,
                                 std::string op_type, json &jobj)
{
    json prelu_data;
    prelu_data["op_type"] = op_type;
    shl_export_json_params_base(params->base, prelu_data);

    // generate attrs
    json attrs;
    attrs["axis"] = params->axis;
    prelu_data["attrs"] = attrs;

    // generate input info
    std::vector<struct csinn_tensor *> in_tensors = {input0, input1};
    shl_export_json_input_tensor(in_tensors, prelu_data);

    // generate output info
    std::vector<struct csinn_tensor *> out_tensors = {output};
    shl_export_json_output_tensor(out_tensors, prelu_data);

    jobj["layers"].push_back(prelu_data);

    return CSINN_TRUE;
}

int shl_export_json_internal(struct csinn_session *sess, char *path)
{
    json jobj;
    struct shl_ref_graph *g = shl_gref_get_graph(sess);

    // generate input names
    jobj["input_names"] = {};
    for (int i = 0; i < g->input_num; i++) {
        jobj["input_names"].push_back(g->input[i]->name);
    }

    // generate output names
    jobj["output_names"] = {};
    for (int i = 0; i < g->output_num; i++) {
        jobj["output_names"].push_back(g->output[i]->name);
    }

    // generate layers
    jobj["layers"] = {};
    for (int i = 0; i < g->layer_index; i++) {
        struct shl_node *node = g->layer[i];
        if (node->type == CSINN_SUBGRAPH) {
            shl_debug_info("There is a subgrah that is ignored temporarily(TODO)\n");
        } else if (node->type >= 0 && node->type < CSINN_OP_SIZE) {
            struct csinn_params_base *params = (struct csinn_params_base *)node->data;
            switch (node->type) {
                case CSINN_OP_CONV2D: {
                    int ret = shl_export_json_conv2d((struct csinn_tensor *)node->in[0]->data,
                                                     (struct csinn_tensor *)node->out[0]->data,
                                                     (struct csinn_tensor *)node->in[1]->data,
                                                     (struct csinn_tensor *)node->in[2]->data,
                                                     (struct csinn_conv2d_params *)params,
                                                     "CSINN_OP_CONV2D", jobj);
                    break;
                }
                case CSINN_OP_DEPTHWISE_CONV2D: {
                    int ret = shl_export_json_conv2d((struct csinn_tensor *)node->in[0]->data,
                                                     (struct csinn_tensor *)node->out[0]->data,
                                                     (struct csinn_tensor *)node->in[1]->data,
                                                     (struct csinn_tensor *)node->in[2]->data,
                                                     (struct csinn_conv2d_params *)params,
                                                     "CSINN_OP_DEPTHWISE_CONV2D", jobj);
                    break;
                }
                case CSINN_OP_GROUP_CONV2D: {
                    int ret = shl_export_json_conv2d((struct csinn_tensor *)node->in[0]->data,
                                                     (struct csinn_tensor *)node->out[0]->data,
                                                     (struct csinn_tensor *)node->in[1]->data,
                                                     (struct csinn_tensor *)node->in[2]->data,
                                                     (struct csinn_conv2d_params *)params,
                                                     "CSINN_OP_GROUP_CONV2D", jobj);
                    break;
                }
                case CSINN_OP_DECONV2D: {
                    int ret = shl_export_json_conv2d((struct csinn_tensor *)node->in[0]->data,
                                                     (struct csinn_tensor *)node->out[0]->data,
                                                     (struct csinn_tensor *)node->in[1]->data,
                                                     (struct csinn_tensor *)node->in[2]->data,
                                                     (struct csinn_conv2d_params *)params,
                                                     "CSINN_OP_DECONV2D", jobj);
                    break;
                }
                case CSINN_OP_DEPTHWISE_DECONV2D: {
                    int ret = shl_export_json_conv2d((struct csinn_tensor *)node->in[0]->data,
                                                     (struct csinn_tensor *)node->out[0]->data,
                                                     (struct csinn_tensor *)node->in[1]->data,
                                                     (struct csinn_tensor *)node->in[2]->data,
                                                     (struct csinn_conv2d_params *)params,
                                                     "CSINN_OP_DEPTHWISE_DECONV2D", jobj);
                    break;
                }
                case CSINN_OP_GROUP_DECONV2D: {
                    int ret = shl_export_json_conv2d((struct csinn_tensor *)node->in[0]->data,
                                                     (struct csinn_tensor *)node->out[0]->data,
                                                     (struct csinn_tensor *)node->in[1]->data,
                                                     (struct csinn_tensor *)node->in[2]->data,
                                                     (struct csinn_conv2d_params *)params,
                                                     "CSINN_OP_GROUP_DECONV2D", jobj);
                    break;
                }
                case CSINN_OP_RELU: {
                    int ret = shl_export_json_siso((struct csinn_tensor *)node->in[0]->data,
                                                   (struct csinn_tensor *)node->out[0]->data,
                                                   params, "CSINN_OP_RELU", jobj);
                    break;
                }
                case CSINN_OP_GLOBAL_AVGPOOL2D: {
                    int ret = shl_export_json_siso((struct csinn_tensor *)node->in[0]->data,
                                                   (struct csinn_tensor *)node->out[0]->data,
                                                   params, "CSINN_OP_GLOBAL_AVGPOOL2D", jobj);
                    break;
                }
                case CSINN_OP_SOFTMAX: {
                    int ret = shl_export_json_softmax((struct csinn_tensor *)node->in[0]->data,
                                                      (struct csinn_tensor *)node->out[0]->data,
                                                      (struct csinn_softmax_params *)params,
                                                      "CSINN_OP_SOFTMAX", jobj);
                    break;
                }
                case CSINN_OP_ADD: {
                    int ret = shl_export_json_diso((struct csinn_tensor *)node->in[0]->data,
                                                   (struct csinn_tensor *)node->in[1]->data,
                                                   (struct csinn_tensor *)node->out[0]->data,
                                                   (struct csinn_diso_params *)params,
                                                   "CSINN_OP_ADD", jobj);
                    break;
                }
                case CSINN_OP_MUL: {
                    int ret = shl_export_json_diso((struct csinn_tensor *)node->in[0]->data,
                                                   (struct csinn_tensor *)node->in[1]->data,
                                                   (struct csinn_tensor *)node->out[0]->data,
                                                   (struct csinn_diso_params *)params,
                                                   "CSINN_OP_MUL", jobj);
                    break;
                }
                case CSINN_OP_MAXPOOL2D: {
                    int ret = shl_export_json_pool2d((struct csinn_tensor *)node->in[0]->data,
                                                     (struct csinn_tensor *)node->out[0]->data,
                                                     (struct csinn_pool_params *)params,
                                                     "CSINN_OP_MAXPOOL2D", jobj);
                    break;
                }
                case CSINN_OP_AVGPOOL2D: {
                    int ret = shl_export_json_pool2d((struct csinn_tensor *)node->in[0]->data,
                                                     (struct csinn_tensor *)node->out[0]->data,
                                                     (struct csinn_pool_params *)params,
                                                     "CSINN_OP_AVGPOOL2D", jobj);
                    break;
                }
                case CSINN_OP_RESHAPE: {
                    int ret = shl_export_json_reshape((struct csinn_tensor *)node->in[0]->data,
                                                      (struct csinn_tensor *)node->out[0]->data,
                                                      (struct csinn_reshape_params *)params,
                                                      "CSINN_OP_RESHAPE", jobj);
                    break;
                }
                case CSINN_OP_FULLYCONNECTED: {
                    int ret = shl_export_json_fullyconnected(
                        (struct csinn_tensor *)node->in[0]->data,
                        (struct csinn_tensor *)node->out[0]->data,
                        (struct csinn_tensor *)node->in[1]->data,
                        (struct csinn_tensor *)node->in[2]->data, (struct csinn_fc_params *)params,
                        "CSINN_OP_FULLYCONNECTED", jobj);
                    break;
                }
                case CSINN_OP_LRN: {
                    int ret = shl_export_json_lrn((struct csinn_tensor *)node->in[0]->data,
                                                  (struct csinn_tensor *)node->out[0]->data,
                                                  (struct csinn_lrn_params *)params, "CSINN_OP_LRN",
                                                  jobj);
                    break;
                }
                case CSINN_OP_CONCAT: {
                    struct csinn_tensor **inputs = (struct csinn_tensor **)shl_mem_alloc(
                        sizeof(struct csinn_tensor *) *
                        ((struct csinn_concat_params *)params)->inputs_count);
                    for (int i = 0; i < ((struct csinn_concat_params *)params)->inputs_count; i++) {
                        inputs[i] = (struct csinn_tensor *)node->in[i]->data;
                    }
                    int ret = shl_export_json_concat(
                        inputs, (struct csinn_tensor *)node->out[0]->data,
                        (struct csinn_concat_params *)params, "CSINN_OP_CONCAT", jobj);
                    shl_mem_free(inputs);
                    break;
                }
                case CSINN_OP_PRELU: {
                    int ret = shl_export_json_prelu((struct csinn_tensor *)node->in[0]->data,
                                                    (struct csinn_tensor *)node->in[1]->data,
                                                    (struct csinn_tensor *)node->out[0]->data,
                                                    (struct csinn_prelu_params *)params,
                                                    "CSINN_OP_PRELU", jobj);
                    break;
                }
                default: {
                    shl_debug_error("unknown op: %d\n", node->type);
                }
            }
        }
    }

    std::ofstream out_file(path);
    out_file << std::setw(2) << jobj << std::endl;

    return CSINN_TRUE;
}

#endif