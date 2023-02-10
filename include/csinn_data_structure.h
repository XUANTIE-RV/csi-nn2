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

/**
 * @file csinn_data_structure.h
 */

#ifndef INCLUDE_CSI_INTERNAL_H_
#define INCLUDE_CSI_INTERNAL_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @defgroup DS Data structure
 * @addtogroup DS
 * @{
 */

/** CSI-NN data type*/
enum csinn_dtype_enum {
    CSINN_DTYPE_BOOL = 0, /**< Boolean */
    CSINN_DTYPE_INT4,     /**< Signed 4 bit fixed-point */
    CSINN_DTYPE_UINT8,    /**< Unsigned 8 bit fixed-point */
    CSINN_DTYPE_INT8,     /**< Signed 8 bit fixed-point */
    CSINN_DTYPE_UINT16,   /**< Unsigned 16 bit fixed-point */
    CSINN_DTYPE_INT16,    /**< Signed 16 bit fixed-point */
    CSINN_DTYPE_UINT32,   /**< Unsigned 32 bit fixed-point */
    CSINN_DTYPE_INT32,    /**< Signed 32 bit fixed-point */
    CSINN_DTYPE_FLOAT16,  /**< Half-precision floating-point */
    CSINN_DTYPE_BFLOAT16, /**< Brain floating-point */
    CSINN_DTYPE_FLOAT32,  /**< Single-precision floating-point */
    CSINN_DTYPE_FLOAT64,  /**< Double-precision floating-point */
    CSINN_DTYPE_INT64,    /**< Signed 64 bit fixed-point */
    CSINN_DTYPE_SIZE,
};

/** CSI-NN data memory type */
enum csinn_mem_type_enum {
    CSINN_MEM_TYPE_CPU_NOT_ALIGNED = 0, /**< Default storage */
    CSINN_MEM_TYPE_CPU_ALIGNED,         /**< Aligned storage */
    CSINN_MEM_TYPE_DMABUF,              /**< DMA buf */
    CSINN_MEM_TYPE_ASP42,               /**< Structed sparsity 4:2 */
    CSINN_MEM_TYPE_ASP41,               /**< Structed sparsity 4:1 */
    CSINN_MEM_TYPE_CPU_ACC,             /**< Accelerator driver or others alloced CPU memory */
};

/** CSI-NN quant type */
enum csinn_quant_enum {
    CSINN_QUANT_UNSET = 0,       /**< The quantization type is not set */
    CSINN_QUANT_INT4_SYM,        /**< Symmetric signed 4-bit fixed-point quantization */
    CSINN_QUANT_UINT8_ASYM,      /**< Asymmetric unsigned 8-bit fixed-point quantization */
    CSINN_QUANT_UINT8_SYM,       /**< Symmetric unsigned 8-bit fixed-point quantization */
    CSINN_QUANT_INT8_ASYM,       /**< Asymmetric signed 8-bit fixed-point quantization */
    CSINN_QUANT_INT8_SYM,        /**< Symmetric signed 8-bit fixed-point quantization */
    CSINN_QUANT_INT16_SYM,       /**< Symmetric signed 16-bit fixed-point quantization */
    CSINN_QUANT_FLOAT16,         /**< 16-bit floating-point quantization */
    CSINN_QUANT_BFLOAT16,        /**< bf16 floating-point quantization */
    CSINN_QUANT_FLOAT32,         /**< 32-bit floating-point not quantized */
    CSINN_QUANT_INT4_ASYM_W_SYM, /**< Signed 4-bit Asymmetric activation and Symmetric weight */
    CSINN_QUANT_INT8_ASYM_W_SYM, /**< Signed 8-bit Asymmetric activation and Symmetric weight */
    CSINN_QUANT_FLOAT16_W_INT8,  /**< 16-bit floating-point and 8-bit symmetric weight */
    CSINN_QUANT_SIZE,
};

/* alias of TH1520, for compatible */
#define CSINN_LIGHT 7

/** CSI-NN API type */
enum csinn_api_enum {
    CSINN_REF = 0,  /**< Reference c */
    CSINN_GREF,     /**< reference graph */
    CSINN_C860,     /**< C860 CPU platform */
    CSINN_C906,     /**< C906 CPU platform */
    CSINN_C920,     /**< C920 CPU platform */
    CSINN_ANOLE,    /**< anole NPU platform */
    CSINN_CH8601,   /**< ch8601 NPU platform */
    CSINN_TH1520,   /**< th1520 NPU platform */
    CSINN_DP1K,     /**< dp1000 NPU platform */
    CSINN_I805,     /**< I805 CPU platform */
    CSINN_E804,     /**< E804 CPU platform */
    CSINN_REF_I805, /**< I805 CPU platform */
    CSINN_C908,     /**< C908 CPU platform */
    CSINN_TVMGEN,   /**< TVM generate platform */
    CSINN_ASP,      /**< ASP platform */
    CSINN_RVV,      /**< RISC-V V extension general platform */
    CSINN_RVM,      /**< RISC-V Matrix extension general platform */
    CSINN_E907,     /**< E907 CPU platform */
    CSINN_API_SIZE,
};

/** CSI-NN run mode */
enum csinn_rmode_enum {
    CSINN_RM_LAYER = 0,       /**< Run by layer */
    CSINN_RM_CPU_GRAPH,       /**< CPU Graph Execution*/
    CSINN_RM_NPU_GRAPH,       /**< NPU Graph Execution */
    CSINN_RM_CPU_BASE_HYBRID, /**< CPU base graph and has subgraph */
    CSINN_RUN_MODE_SIZE,
};

/** CSI-NN model save */
enum csinn_mode_save_enum {
    CSINN_SAVE_AND_RUN = 0, /**< Save the model and run it */
    CSINN_SAVE_ONLY,        /**< Save the model only */
    CSINN_RUN_ONLY,         /**< Run the model only */
};

/** CSI-NN OP and utils */
enum csinn_op_enum {
    CSINN_OP_ABS = 0,
    CSINN_OP_ACOS,
    CSINN_OP_ACOSH,
    CSINN_OP_ADD,
    CSINN_OP_ALL,
    CSINN_OP_AND,
    CSINN_OP_ANY,
    CSINN_OP_ARANGE,
    CSINN_OP_ARGMAX,
    CSINN_OP_ARGMIN,
    CSINN_OP_ASIN,
    CSINN_OP_ASINH,
    CSINN_OP_ATAN,
    CSINN_OP_ATANH,
    CSINN_OP_AVGPOOL2D,
    CSINN_OP_AVGPOOL3D,
    CSINN_OP_BN,
    CSINN_OP_BATCH_TO_SPACE,
    CSINN_OP_BATCH_TO_SPACE_ND,
    CSINN_OP_BROADCOST,
    CSINN_OP_CACHE_MATMUL,
    CSINN_OP_CACHE_CONV1D,
    CSINN_OP_CAST,
    CSINN_OP_CEIL,
    CSINN_OP_CLIP,
    CSINN_OP_COL2IM,
    CSINN_OP_CONCAT,
    CSINN_OP_CONV1D,
    CSINN_OP_CONV2D,
    CSINN_OP_CONV2D_RELU,
    CSINN_OP_CONV2D_RELU6,
    CSINN_OP_CONV2D_CHANNEL,
    CSINN_OP_CONV2D_CHANNEL_RELU,
    CSINN_OP_CONV2D_CHANNEL_RELU6,
    CSINN_OP_DEPTHWISE_CONV1D,
    CSINN_OP_DEPTHWISE_CONV2D,
    CSINN_OP_DEPTHWISE_CONV2D_RELU,
    CSINN_OP_DEPTHWISE_CONV2D_RELU6,
    CSINN_OP_DEPTHWISE_CONV2D_CHANNEL,
    CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU,
    CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6,
    CSINN_OP_GROUP_CONV1D,
    CSINN_OP_GROUP_CONV2D,
    CSINN_OP_GROUP_CONV2D_RELU,
    CSINN_OP_GROUP_CONV2D_RELU6,
    CSINN_OP_GROUP_CONV2D_CHANNEL,
    CSINN_OP_GROUP_CONV2D_CHANNEL_RELU,
    CSINN_OP_CONV3D,
    CSINN_OP_DATA_CONVERT,
    CSINN_OP_COS,
    CSINN_OP_COSH,
    CSINN_OP_CROP,
    CSINN_OP_CUMPROD,
    CSINN_OP_CUMSUM,
    CSINN_OP_DECONV2D,
    CSINN_OP_DEPTHWISE_DECONV2D,
    CSINN_OP_GROUP_DECONV2D,
    CSINN_OP_DECONV3D,
    CSINN_OP_DEPTH_TO_SPACE,
    CSINN_OP_DIV,
    CSINN_OP_ELU,
    CSINN_OP_EQUANL,
    CSINN_OP_ERF,
    CSINN_OP_EXP,
    CSINN_OP_EXPAND_DIMS,
    CSINN_OP_EXPM1,
    CSINN_OP_FLATTEN,
    CSINN_OP_FLOOR_DIVIDE,
    CSINN_OP_FLOOR_MOD,
    CSINN_OP_FLOOR,
    CSINN_OP_FSMN,
    CSINN_OP_FULLYCONNECTED,
    CSINN_OP_GATHER_ND,
    CSINN_OP_GATHER,
    CSINN_OP_GLOBAL_AVGPOOL2D,
    CSINN_OP_GLOBAL_MAXPOOL2D,
    CSINN_OP_GREATHER_EQUAL,
    CSINN_OP_GREATHER,
    CSINN_OP_HARD_SIGMOID,
    CSINN_OP_IM2COL,
    CSINN_OP_ISNAN,
    CSINN_OP_L2N,
    CSINN_OP_L2POOL2D,
    CSINN_OP_LAYER_NORM,
    CSINN_OP_LEAKY_RELU,
    CSINN_OP_LESS_EQUAL,
    CSINN_OP_LESS,
    CSINN_OP_LOG_SOFTMAX,
    CSINN_OP_LOG,
    CSINN_OP_LOG1P,
    CSINN_OP_LOGICAL_AND,
    CSINN_OP_LOGICAL_NOT,
    CSINN_OP_LOGICAL_OR,
    CSINN_OP_LOGICAL_XOR,
    CSINN_OP_LRN,
    CSINN_OP_MATMUL,
    CSINN_OP_MAX,
    CSINN_OP_MAXIMUM,
    CSINN_OP_MAXPOOL2D,
    CSINN_OP_MAXPOOL2D_LOCAT,
    CSINN_OP_MAXPOOL3D,
    CSINN_OP_MEAN,
    CSINN_OP_MEAN_STRIDE,
    CSINN_OP_MIN,
    CSINN_OP_MIN_STRIDE,
    CSINN_OP_MINIMUM,
    CSINN_OP_MOD,
    CSINN_OP_MUL,
    CSINN_OP_NDARRAY_SIZE,
    CSINN_OP_NEGATIVE,
    CSINN_OP_NON_MAX_SUPPRESSION,
    CSINN_OP_NOT_EQUAL,
    CSINN_OP_NOT,
    CSINN_OP_ONE_HOT,
    CSINN_OP_OR,
    CSINN_OP_PAD,
    CSINN_OP_POWER,
    CSINN_OP_PRELU,
    CSINN_OP_PROD,
    CSINN_OP_PROPOSAL,
    CSINN_OP_PSROIPOOLING,
    CSINN_OP_REDUCE_LOGSUMEXP,
    CSINN_OP_REDUCE_MAX,
    CSINN_OP_REDUCE_MEAN,
    CSINN_OP_REDUCE_MIN,
    CSINN_OP_REDUCE_PROD,
    CSINN_OP_REDUCE_SUM,
    CSINN_OP_RELU,
    CSINN_OP_RELU1,
    CSINN_OP_RELU6,
    CSINN_OP_RELUN,
    CSINN_OP_REORG,
    CSINN_OP_RESHAPE,
    CSINN_OP_RESIZE,
    CSINN_OP_REVERSE,
    CSINN_OP_ROIALIGN,
    CSINN_OP_ROIPOOL,
    CSINN_OP_ROUND,
    CSINN_OP_RSQRT,
    CSINN_OP_SCATTER_ND,
    CSINN_OP_SEGMENT_MAX,
    CSINN_OP_UNSORTED_SEGMENT_MAX,
    CSINN_OP_SEGMENT_MEAN,
    CSINN_OP_UNSORTED_SEGMENT_MEAN,
    CSINN_OP_SEGMENT_MIN,
    CSINN_OP_UNSORTED_SEGMENT_MIN,
    CSINN_OP_SEGMENT_PROD,
    CSINN_OP_UNSORTED_SEGMENT_PROD,
    CSINN_OP_SEGMENT_SUM,
    CSINN_OP_UNSORTED_SEGMENT_SUM,
    CSINN_OP_SELECT,
    CSINN_OP_SEQUENCE_MASK,
    CSINN_OP_SHAPE,
    CSINN_OP_SHUFFLE_CHANNEL,
    CSINN_OP_SIGMOID,
    CSINN_OP_SIGN,
    CSINN_OP_SIN,
    CSINN_OP_SINH,
    CSINN_OP_SLICE,
    CSINN_OP_SOFTMAX,
    CSINN_OP_SOFTPLUS,
    CSINN_OP_SOFTRELU,
    CSINN_OP_SOFTSIGN,
    CSINN_OP_SPACE_TO_BATCH,
    CSINN_OP_SPACE_TO_BATCH_ND,
    CSINN_OP_SPACE_TO_DEPTH,
    CSINN_OP_SPLIT,
    CSINN_OP_SQRT,
    CSINN_OP_SQUARE,
    CSINN_OP_SQUEEZE,
    CSINN_OP_STACK,
    CSINN_OP_STRIDED_SLICE,
    CSINN_OP_SUB,
    CSINN_OP_SUM,
    CSINN_OP_TAN,
    CSINN_OP_TANH,
    CSINN_OP_THRESHOLD_RELU,
    CSINN_OP_TILE,
    CSINN_OP_TOPK,
    CSINN_OP_TRANSPOSE,
    CSINN_OP_TRUNC,
    CSINN_OP_UNPOOLING,
    CSINN_OP_UNSTACK,
    CSINN_OP_WHERE,
    CSINN_OP_WHERE_SOFTMAX,
    CSINN_OP_XOR,
    CSINN_OP_YUV_RGB_SCALE,
    CSINN_OP_INSTANCE_NORM,

    CSINN_OP_SIZE,

    /* graph */
    CSINN_TENSOR,
    CSINN_SUBGRAPH,
    CSINN_SUBGRAPH_RETURN,
    CSINN_OP_AND_UTILS_SIZE,
};

enum csinn_runtime_enum {
    CSINN_SESSION_INIT,
    CSINN_SESSION_DEINIT,
    CSINN_SESSION_SETUP,
    CSINN_SESSION_RUN,
    CSINN_UPDATE_INPUT,
    CSINN_UPDATE_OUTPUT,
    CSINN_SET_INPUT_NUMBER,
    CSINN_SET_OUTPUT_NUMBER,
    CSINN_GET_INPUT_NUMBER,
    CSINN_GET_OUTPUT_NUMBER,
    CSINN_SET_INPUT,
    CSINN_SET_OUTPUT,
    CSINN_GET_INPUT,
    CSINN_GET_OUTPUT,
    CSINN_TENSOR_ENTRY,
    CSINN_LOAD_BG,
    CSINN_RUNTIME_OP_SIZE,
};

/** CSI-NN convolution mode */
enum csinn_conv_mode_enum {
    CSINN_DIRECT = 0x0,   /**< using direct optimizational convolution */
    CSINN_WINOGRAD = 0x1, /**< using winograd fast convolution */
    CSINN_GEMM = 0x2,     /**< using im2col + gemm convolution, im2col is optional */
};

/** CSI-NN pad mode */
enum csinn_pad_enum {
    CSINN_PAD_CONSTANT = 0x0, /**< pads with constant_value pad_value */
    CSINN_PAD_EDGE = 0x1,     /**< pads using the edge values of the input array */
    CSINN_PAD_REFLECT = 0x2,  /**< pads by reflecting values with respect to the edge */
};

/** CSI-NN resize mode */
enum csinn_resize_enum {
    CSINN_RESIZE_BILINEAR = 0x0,         /**< Bilinear interpolation */
    CSINN_RESIZE_NEAREST_NEIGHBOR = 0x1, /**< Nearest neighbor interpolation */
    CSINN_RESIZE_NEAREST_BICUBIC = 0x2,  /**< Bicubic interpolation */
};

/** CSI-NN depth2space mode */
enum csinn_depth2space_enum {
    CSINN_DEPTHTOSPACE_DCR = 0x0, /**< arranges data in the order of (depth,column,row) */
    CSINN_DEPTHTOSPACE_CRD = 0x1, /**< arranges data in the order of (column,row,depth) */
};

/** CSI-NN LRN type */
enum csinn_lrn_enum {
    CSINN_LRN_ACROSS_CHANNELS = 0x0, /**< local response normalization across channels/channels */
    CSINN_LRN_WITHIN_CHANNEL,        /**< local response normalization within the same channel */
};

/** CSI-NN layout type */
enum csinn_layout_enum {
    CSINN_LAYOUT_NULL = 0x0, /**< Not set */
    // NCHW
    // ACTIVITION
    CSINN_LAYOUT_N,     /**< NCHW input and output, 1 dimension */
    CSINN_LAYOUT_NC,    /**< NCHW input and output, 2 dimensions */
    CSINN_LAYOUT_NCW,   /**< NCHW input and output, 3 dimensions */
    CSINN_LAYOUT_NCHW,  /**< NCHW input and output, 4 dimensions */
    CSINN_LAYOUT_NCDHW, /**< NCHW input and output, 5 dimensions */
    // WEIGHT
    CSINN_LAYOUT_O,      /**< NCHW constant, 1 dimension */
    CSINN_LAYOUT_OI,     /**< NCHW constant, 2 dimensions */
    CSINN_LAYOUT_O16I16, /**< 16 bytes in parallel for ASP platform */
    CSINN_LAYOUT_O32I32, /**< 32 bytes in parallel for ASP platform */
    CSINN_LAYOUT_OIW,    /**< NCHW constant, 3 dimension */
    CSINN_LAYOUT_OIHW,   /**< NCHW constant, 4 dimension */
    CSINN_LAYOUT_OIDHW,  /**< NCHW constant, 5 dimension */
    CSINN_LAYOUT_O1HW,   /**< NCHW constant, depthwise convolution only */

    // NHWC
    // ACTIVITION
    CSINN_LAYOUT_NWC,   /**< NHWC input and output, 3 dimensions */
    CSINN_LAYOUT_NHWC,  /**< NHWC input and output, 4 dimensions */
    CSINN_LAYOUT_NDHWC, /**< NHWC input and output, 5 dimensions */
    // WEIGHT
    CSINN_LAYOUT_OWI,      /**< NHWC constant, 3 dimensions */
    CSINN_LAYOUT_OHWI,     /**< NHWC constant, 4 dimensions */
    CSINN_LAYOUT_O16HWI16, /**< 16 bytes in parallel for ASP platform */
    CSINN_LAYOUT_O32HWI32, /**< 32 bytes in parallel for ASP platform */
    CSINN_LAYOUT_ODHWI,    /**< NHWC constant, 5 dimensions */
    CSINN_LAYOUT_1HWO,     /**< NHWC constant, depthwise convolution only */
    CSINN_LAYOUT_1HW16O16, /**< 16 bytes in parallel for ASP platform */
    CSINN_LAYOUT_1HW32O32, /**< 32 bytes in parallel for ASP platform */

    // NC1HWC0
    // ACTIVITION
    // RVV optimization format: c0=4/8/8 for fp32/fp16/int8 when vlen=128
    CSINN_LAYOUT_NC1C0,    /**< NC1HWC0 input and output, 2 dimension */
    CSINN_LAYOUT_NC1WC0,   /**< NC1HWC0 input and output, 3 dimension */
    CSINN_LAYOUT_NC1HWC0,  /**< NC1HWC0 input and output, 4 dimension */
    CSINN_LAYOUT_NC1DHWC0, /**< NC1HWC0 input and output, 5 dimension */

    // for 6D shape
    CSINN_LAYOUT_NLCDHW, /**< NCHW input and output, 6 dimensions */
};

/** CSI-NN return type */
enum csinn_status_enum {
    CSINN_UNSUPPORT_LAYOUT = -3, /**< An error occurred while executing the function.
                                      An unsupported layout is used */
    CSINN_UNSUPPORT_DTYPE = -2,  /**< An error occurred while executing the function.
                                      An unsupported data type is used */
    CSINN_CALLBACK_UNSET = -1,   /**< An error occurred while executing the function.
                                      The callback function is not set */
    CSINN_FALSE = 0,             /**< An error occurred while executing the function */
    CSINN_TRUE = 1,              /**< The function runs successfully */
};

/** CSI-NN optimize level */
enum csinn_optimize_method_enum {
    CSINN_OPT_FORCE_REPLACE = -1,    /**< force replace, must choose it */
    CSINN_OPT_ASM = 10,              /**< assmbly optimized */
    CSINN_OPT_INTRINSIC = 20,        /**< Intrinsic code optimized */
    CSINN_OPT_TVMGEN = 100,          /**< generate by HHB */
    CSINN_OPT_C_REFERENCE = 1000,    /**< C reference code that have not optimized */
    CSINN_OPT_UNSUPPORTED = 1000000, /**< Cannot supported */
};

/** CSI-NN proifle type */
enum csinn_profiler_enum {
    CSINN_PROFILER_LEVEL_UNSET = 0, /**< The performance analysis mode is not set */
    CSINN_PROFILER_LEVEL_TIMER,     /**< The performance analysis mode, which prints some time
                                         information */
    CSINN_PROFILER_LEVEL_DUMP,      /**< The performance analysis mode, which dump the
                                         the output tensor value of every layer. */
    CSINN_PROFILER_LEVEL_ALL,       /**< The performance analysis mode, do all operations that
                                         mentioned above. */
};

/** debug type */
enum csinn_debug_enum {
    CSINN_DEBUG_LEVEL_DEBUG = -2, /**< developer debugging level */
    CSINN_DEBUG_LEVEL_INFO,       /**< details level */
    CSINN_DEBUG_LEVEL_WARNING,    /**< warning message */
    CSINN_DEBUG_LEVEL_ERROR,      /**< error message */
    CSINN_DEBUG_LEVEL_FATAL,      /**< program crash */
};

/** CSI-NN quantization information */
struct csinn_quant_info {
    int32_t zero_point; /**< Zero point value */
    float scale;        /**< Scale value */
    int32_t multiplier; /**< Multiplier value, compose scale with shift */
    int32_t shift;      /**< Shift value, compose scale with multiplier */
    float min;          /**< Minimum of tensor values */
    float max;          /**< Maximum of tensor values */
};

#define MAX_DIM 8
/** CSI-NN tensor */
struct csinn_tensor {
    void *data;                     /**< Real data pointing to tensors */
    enum csinn_dtype_enum dtype;    /**< Description the data type of the tensor */
    enum csinn_mem_type_enum mtype; /**< Describes the storage type of the tensor */
    int32_t dim[MAX_DIM];           /**< Describes the size of each dimension in the tensor.  */
    int32_t dim_count;              /**< The number of tensor dimensions. The current version
                                         supports a maximum of eight dimensions */
    uint32_t is_const;              /**< Whether the marker tensor is a constant */
    char *name;                     /**< The name of the current tensor */
    int32_t layout;                 /**< Describes the data layout type of the tensor */
    int32_t quant_channel; /**< Specifies the number of qinfo. 0 indicates non-quantization,
                                1 indicates normal quantization, and greater than 1 indicates
                                channel quantization */
    struct csinn_quant_info *qinfo; /**< An array pointing to the quantization information */
    struct csinn_session *sess;     /**< Indicates the data structure of the current session */
};

/** CSI-NN model */
struct csinn_model {
    char *bm_path;     /**< The path of the model. Select one from the address of the model */
    void *bm_addr;     /**< The address of the model. Select one from the path of the model */
    size_t bm_size;    /**< The size of the model, which is used with the model address */
    int32_t save_mode; /**< Save mode */
    int32_t priority;  /**< The priority of model execution */
};

/** CSI-NN session */
struct csinn_session {
    int32_t base_dtype;                    /**< The basic data type, which is used as the default
                                                data type for subsequent tensor allocation */
    int32_t base_layout;                   /**< The basic data layout is used as the default data
                                                layout for subsequent tensor allocation */
    int32_t base_api;                      /**< The basic operator type, which is used as the basic
                                                attribute of subsequent operators */
    int32_t base_run_mode;                 /**< The basic execution mode of subsequent operators */
    enum csinn_quant_enum base_quant_type; /**< Basic quantization type */
    struct csinn_model model;              /**< Model information */
    int32_t debug_level;                   /**< Debugging level */
    int32_t profiler_level;                /**< Performance analysis level */
    int32_t input_num;                     /**< The number of input */
    int32_t output_num;                    /**< The number of output */
    struct csinn_tensor **input;           /**< Point to all inputs */
    struct csinn_tensor **output;          /**< Point to all outputs */
    void *td;                              /**< Refers to private data, which can generally point to
                                                the structure representing the graph in the driver */
    bool dynamic_shape;                    /**< Wether to infer shape */
};

/** CSI-NN tensor */
struct csinn_callback {
    int (*init)(); /**< initialization */
    int (*est)();  /**< establish graph */
    int (*exec)(); /**< execute real compute */
    int (*caps)(); /**< capabilities */
    int (*perf)(); /**< profiling */
};

/** CSI-NN params base */
struct csinn_params_base {
    struct csinn_callback *cb;        /**< The callback function pointing to the operator */
    char *name;                       /**< The name of the operator */
    int32_t layout;                   /**< The data layout that is suitable for calculation */
    int32_t api;                      /**< Tag different platform interfaces */
    enum csinn_quant_enum quant_type; /**< The quantization type of the operator */
    struct csinn_session *sess;       /**< Indicates the data structure of the current session */
};

/** CSI-NN fsmn params */
struct csinn_fsmn_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t l_order;               /**< Number of frames in the past (l_order-1) */
    int32_t r_order;               /**< The number of future frames to be computed */
    int32_t l_stride; /**< The sampling frequency of past frames involved in the calculation */
    int32_t r_stride; /**< The sampling frequency of future frames involved in the calculation */
    int32_t unavailable_frames; /**< The number of invalid frames */
};

/** CSI-NN conv2d params */
struct csinn_conv2d_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t group;                 /**< The number of convolutional groups */
    int32_t stride_height;         /**< Vertical step */
    int32_t stride_width;          /**< Horizontal step */
    int32_t pad_top;               /**< The number of top padding */
    int32_t pad_left;              /**< The number of left padding */
    int32_t pad_down;              /**< The number of bottom padding */
    int32_t pad_right;             /**< The number of right padding */
    int32_t dilation_height;       /**< Longitudinal expansion coefficient */
    int32_t dilation_width;        /**< Horizontal expansion coefficient */
    int32_t out_pad_height;
    int32_t out_pad_width;
    struct {
        struct csinn_tensor *kernel_tm;
        enum csinn_conv_mode_enum conv_mode;
        int32_t fuse_zp2bias;
    } conv_extra; /**< The structure used for cpu convolution optimization, including intermediate
                       conversion weight and convolution type (gemm/winograd) */
};

/** CSI-NN conv3d params */
struct csinn_conv3d_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t group;                 /**< The number of convolutional groups */
    int32_t stride_depth;          /**< Step in the depth direction */
    int32_t stride_height;         /**< Vertical step */
    int32_t stride_width;          /**< Horizontal step */
    int32_t pad_top;               /**< The number of top padding */
    int32_t pad_left;              /**< The number of left padding */
    int32_t pad_down;              /**< The number of bottom padding */
    int32_t pad_right;             /**< The number of right padding */
    int32_t pad_front;             /**< The number of front padding */
    int32_t pad_back;              /**< The number of back padding */
    int32_t dilation_depth;        /**< Expansion coefficient in depth direction  */
    int32_t dilation_height;       /**< Longitudinal expansion coefficient */
    int32_t dilation_width;        /**< Horizontal expansion coefficient */
    int32_t out_pad_depth;         /**<  */
    int32_t out_pad_height;        /**<  */
    int32_t out_pad_width;         /**<  */
};

/** CSI-NN fullyconnected params */
struct csinn_fc_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t units;                 /**< The number of output nodes at the current layer */
    struct {
        int32_t fuse_zp2bias;
    } fc_extra;
};

/** CSI-NN pooling params */
struct csinn_pool_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t pool_type;             /**< Pool type */
    int32_t filter_height;         /**< Pool height */
    int32_t filter_width;          /**< Pool width */
    int32_t filter_depth;          /**< Pool depth */
    int32_t stride_height;         /**< Vertical step */
    int32_t stride_width;          /**< Horizontal step */
    int32_t stride_depth;          /**< Step in the depth direction */
    int32_t pad_top;               /**< The number of top padding */
    int32_t pad_left;              /**< The number of left padding */
    int32_t pad_down;              /**< The number of bottom padding */
    int32_t pad_right;             /**< The number of right padding */
    int32_t pad_front;             /**< The number of front padding */
    int32_t pad_back;              /**< The number of back padding */
    int32_t ceil_mode;             /**< When True, will use ceil instead of floor to
                                        compute the output shape */
    bool count_include_pad;        /**< Specifies whether to include the edge pad during pooling,
                                        which usually occurs in mean pooling */
};

/** CSI-NN unpooling params */
struct csinn_unpooling_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t scale_height;          /**< Vertical scale */
    int32_t scale_width;           /**< Horizontal scale */
    int32_t pad_out_height;        /**< Vertical padding */
    int32_t pad_out_width;         /**< Horizontal padding */
};

/** CSI-NN roi_align params */
struct csinn_roi_align_params {
    struct csinn_params_base base;    /**< The basic information of the operator */
    int32_t pooled_size_h;            /**< Vertical height of pooling core */
    int32_t pooled_size_w;            /**< Horizontal height of pooling core */
    float spatial_scale;              /**< roi_align separate scaling for calculation */
    int32_t spatial_scale_multiplier; /**< The multiplier used for fixed-point calculation, together
                                           with spatial_scale_shift to form spatial_scale */
    int32_t spatial_scale_shift;      /**< Number of shifts, together with spatial_scale_multiplier
                                           to form spatial_scale */
    int32_t sample_ratio;             /**< Interpolation proportional coefficient */
};

/** CSI-NN roi_pool params */
struct csinn_roi_pool_params {
    struct csinn_params_base base;    /**< The basic information of the operator */
    int32_t pooled_size_h;            /**< Vertical height of pooling core */
    int32_t pooled_size_w;            /**< Horizontal height of pooling core */
    float spatial_scale;              /**< roi_pool separate scaling for calculation */
    int32_t spatial_scale_multiplier; /**< The multiplier used for fixed-point calculation, together
                                           with spatial_scale_shift to form spatial_scale */
    int32_t spatial_scale_shift;      /**< Number of shifts, together with spatial_scale_multiplier
                                           to form spatial_scale */
};

/** CSI-NN single input single output params */
struct csinn_siso_params {
    struct csinn_params_base base; /**< The basic information of the operator */
};

/** CSI-NN scatter_nd params */
struct csinn_scatter_nd_params {
    struct csinn_params_base base; /**< The basic information of the operator */
};

/** CSI-NN sigmoid params */
struct csinn_sigmoid_params {
    struct csinn_params_base base; /**< The basic information of the operator */
};

/** CSI-NN relu params */
struct csinn_relu_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    float n;              /**< Coefficients used by variants of ReLU such as n/alpha/threshold */
    int32_t n_multiplier; /**< The multiplier used in decimal point calculation, together with
                               n_shift to form n */
    int32_t n_shift;      /**< Used for fixed-point calculation number of shifts,together with
                               n_multiplier to form n */
};

/** CSI-NN prelu params */
struct csinn_prelu_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t axis;                  /**< The axis used to calculate prelu */
};

/** CSI-NN softmax params */
struct csinn_softmax_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t axis;                  /**< The axis used to calculate softmax */
};

/** CSI-NN where softmax params */
struct csinn_where_softmax_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t axis;                  /**< The axis used to calculate softmax */
    float minus_inf;               /**< Constant attribution minus inf */
};

/** CSI-NN bn params */
struct csinn_bn_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    float epsilon;                 /**< Batch normalization calculation of epsilon coefficient */
    int32_t epsilon_multiplier;    /**< The multiplier used in decimal point calculation, together
                                        with epsilon_shift to form epsilon */
    int32_t epsilon_shift;         /**< used for fixed-point calculation number of shifts, together
                                        with epsilon_multiplier to form epsilon */
};

/** CSI-NN l2n params */
struct csinn_l2n_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    float epsilon;                 /**< Batch normalization calculation of epsilon coefficient */
    int32_t epsilon_multiplier;    /**< The multiplier used in decimal point calculation, together
                                        with epsilon_shift to form epsilon */
    int32_t epsilon_shift;         /**< used for fixed-point calculation number of shifts, together
                                        with epsilon_multiplier to form epsilon */
    int32_t *axis;                 /**< Axis */
    int32_t n;                     /**< Number of axes */
};

/** CSI-NN lrn params */
struct csinn_lrn_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t range;                 /**< Compute the number of channels used */
    double bias;                   /**< bias coefficient */
    int32_t bias_multiplier;       /**< The multiplier used for decimal point calculation, together
                                        with bias_shift to form bias */
    int32_t bias_shift;            /**< The shift used for decimal point calculation */
    double alpha;                  /**< alpha coefficient */
    int32_t alpha_multiplier;      /**< The multiplier used for decimal point calculation, together
                                        with alpha_shift to form alpha */
    int32_t alpha_shift;           /**< The shift used for decimal point calculation */
    double beta;                   /**< beta coefficient */
    int32_t beta_multiplier;       /**< The multiplier used for decimal point calculation, together
                                        with beta_shift to form beta */
    int32_t beta_shift;            /**< The shift used for decimal point calculation */
    enum csinn_lrn_enum norm_region;
};

/** CSI-NN matmul params */
struct csinn_matmul_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    bool trans_a;                  /**< Indicates whether the first input is transposed */
    bool trans_b;                  /**< Indicates whether the second input is transposed */
};

/** CSI-NN double input single output params */
struct csinn_diso_params {
    struct csinn_params_base base; /**< The basic information of the operator */
};

/** CSI-NN select params */
struct csinn_select_params {
    struct csinn_params_base base; /**< The basic information of the operator */
};

/** CSI-NN pad params */
struct csinn_pad_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *pad_before;           /**< The number of front padding */
    int32_t *pad_after;            /**< The number of back padding */
    int32_t pad_num;               /**< Pad length */
    float pad_value;               /**< Pad value */
    enum csinn_pad_enum pad_mode;  /**< Pad mode */
};

/** CSI-NN resize params */
struct csinn_resize_params {
    struct csinn_params_base base;      /**< The basic information of the operator */
    enum csinn_resize_enum resize_mode; /**< Resize mode */
    bool align_corners;                 /**< Align corners */
};

/** CSI-NN concat params */
struct csinn_concat_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t inputs_count;          /**< The number of inputs */
    int32_t axis;                  /**< Sliced axis */
};

/** CSI-NN proposal params */
struct csinn_proposal_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    float *scales;                 /**< The scale parameter used for proposal calculation */
    int32_t *scale_multipliers;    /**< The multiplier used for fixed-point calculation */
    int32_t *scale_shifts;         /**< The number of shifts used in the fixed-point calculation */
    int32_t scales_num;            /**< The number of scale parameters */
    float *ratios;                 /**< The ratio parameter used for proposal calculation */
    int32_t *ratio_multipliers;    /**< The multiplier used for fixed-point calculation */
    int32_t *ratio_shifts;         /**< The number of shifts used in the fixed-point calculation */
    int32_t ratios_num;            /**< Mumber of ratio parameters */
    int32_t feature_stride;        /**< Feature step */
    float threshold;               /**< The threshold parameter used for proposal calculation */
    int32_t threshold_multiplier;  /**< The multiplier used for fixed-point calculation */
    int32_t threshold_shift;       /**< The number of shifts used in the fixed-point calculation */
    int rpn_pre_nms_top_n;         /**< nms post-processing top */
    int rpn_post_nms_top_n;        /**< nms pre-processing top */
    int rpn_min_size;              /**< rpn minimum */
    bool iou_loss;                 /**< iou loss */
};

/** CSI-NN psroipooling params */
struct csinn_psroipooling_params {
    struct csinn_params_base base;    /**< The basic information of the operator */
    int32_t output_dim;               /**< Output dimension */
    int32_t group_size;               /**< Number of groups */
    float spatial_scale;              /**< psroipooling separate scaling for calculation */
    int32_t spatial_scale_multiplier; /**< The multiplier used for fixed-point calculation */
    int32_t spatial_scale_shift;      /**< Number of shifts */
};

/** CSI-NN transpose params */
struct csinn_transpose_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *permute;              /**< Order of output dimensions */
    int32_t permute_num;           /**< Size of output dimensions */
};

/** CSI-NN reshape params */
struct csinn_reshape_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *shape;                /**< The output dimension after reshape */
    int32_t shape_num;             /**< The length of the reshape */
};

/** CSI-NN shape params */
struct csinn_shape_params {
    struct csinn_params_base base; /**< The basic information of the operator */
};

/** CSI-NN expand_dims params */
struct csinn_expand_dims_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t axis;                  /**< Axis */
};

/** CSI-NN reverse params */
struct csinn_reverse_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t axis;                  /**< Axis */
};

/** CSI-NN flatten params */
struct csinn_flatten_params {
    struct csinn_params_base base; /**< The basic information of the operator */
};

/** CSI-NN crop params */
struct csinn_crop_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t axis;                  /**< Axis */
    int32_t *offset;               /**< Offset value */
    int32_t offset_num;            /**< The length of the offset */
};

/** CSI-NN slice params */
struct csinn_slice_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *begin;                /**< Begin */
    int32_t *end;                  /**< End */
    int32_t *strides;              /**< Step */
    int32_t slice_num;             /**< Indicates the dim length of the slice */
};

/** CSI-NN split params */
struct csinn_split_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *split_index;          /**< The split sequence */
    int32_t output_num;            /**< The number of outputs */
    int32_t axis;                  /**< Axis */
};

/** CSI-NN stack params */
struct csinn_stack_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t inputs_count;          /**< Input number */
    int32_t axis;                  /**< Axis */
};

/** CSI-NN tile params */
struct csinn_tile_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *reps;                 /**< Number of duplicates */
    int32_t reps_num;              /**< The number of repeated latitudes */
};

/** CSI-NN arange params */
struct csinn_arange_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    float start;                   /**< Start position */
    int32_t start_multiplier;      /**< The multiplier used for decimal point calculation */
    int32_t start_shift;           /**< The shift used for decimal point calculation */
    float stop;                    /**< End position */
    int32_t stop_multiplier;       /**< The multiplier used for decimal point calculation */
    int32_t stop_shift;            /**< The shift used for decimal point calculation */
    float step;                    /**< Step */
    int32_t step_multiplier;       /**< The multiplier used for decimal point calculation */
    int32_t step_shift;            /**< The shift used for decimal point calculation */
};

/** CSI-NN where params */
struct csinn_where_params {
    struct csinn_params_base base; /**< The basic information of the operator */
};

/** CSI-NN unstack params */
struct csinn_unstack_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t outputs_count;         /**< Output number */
    int32_t axis;                  /**< Axis */
};

/** CSI-NN gather params */
struct csinn_gather_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t axis;                  /**< Gather indices axis */
};
/** CSI-NN gather_nd params */
struct csinn_gather_nd_params {
    struct csinn_params_base base; /**< The basic information of the operator */
};

/** CSI-NN squeeze params */
struct csinn_squeeze_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *axis;                 /**< Axis */
    int32_t axis_num;              /**< The length of the axis */
};

/** CSI-NN ndarray_size params */
struct csinn_ndarray_size_params {
    struct csinn_params_base base; /**< The basic information of the operator */
};

/** CSI-NN space_to_batch params */
struct csinn_space_to_batch_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t pad_top;               /**< The number of top padding */
    int32_t pad_bottom;            /**< The number of bottom padding */
    int32_t pad_left;              /**< The number of left padding */
    int32_t pad_right;             /**< The number of right padding */
    int32_t block_size;            /**< Block size */
};

/** CSI-NN space_to_batch_nd params */
struct csinn_space_to_batch_nd_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *paddings;             /**< The number of padding points to each spatial dimension */
    int32_t *block_shape;          /**< Indicates the block size of each dimension */
    int32_t spatial_dim_cnt;       /**< The size of the spatial dimension */
};

/** CSI-NN batch_to_space params */
struct csinn_batch_to_space_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t crop_top;              /**< The number of cropings at the top */
    int32_t crop_bottom;           /**< The number of cropings at the bottom */
    int32_t crop_left;             /**< The number of cropings at the left */
    int32_t crop_right;            /**< The number of cropings at the right */
    int32_t block_size;            /**< Block size */
};

/** CSI-NN batch_to_space_nd params */
struct csinn_batch_to_space_nd_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *crops;                /**< Indicates the number of crops to be cropped in each spatial
                                        dimension */
    int32_t *block_shape;          /**< Indicates the block size of each dimension */
    int32_t spatial_dim_cnt;       /**< The size of the spatial dimension */
};

/** CSI-NN space_to_depth params */
struct csinn_space_to_depth_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t block_size;            /**< Block size */
};

/** CSI-NN depth_to_space params */
struct csinn_depth_to_space_params {
    struct csinn_params_base base;    /**< The basic information of the operator */
    enum csinn_depth2space_enum mode; /**< depth_to_space mode */
    int32_t block_size;               /**< Block size */
};

/** CSI-NN one_hot params */
struct csinn_one_hot_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    float f_on_value;              /**< float on value */
    float f_off_value;             /**< float off value */
    int32_t on_value;              /**< int on value */
    int32_t off_value;             /**< int off value */
    int32_t depth;                 /**< The size of the output tensor */
    int32_t axis;                  /**< Axis */
};

/** CSI-NN sequence_mask params */
struct csinn_sequence_mask_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    float mask_value;              /**< Mask value */
    int32_t mask_value_multiplier; /**< Multiplier */
    int32_t mask_value_shift;      /**< Shift */
    int32_t axis;                  /**< Axis */
};

/** CSI-NN im2col params */
struct csinn_im2col_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t pad_top;               /**< The number of top padding */
    int32_t pad_down;              /**< The number of bottom padding */
    int32_t pad_left;              /**< The number of left padding */
    int32_t pad_right;             /**< The number of right padding */
    int32_t stride_h;              /**< Vertical step */
    int32_t stride_w;              /**< Horizontal step */
    int32_t kernel_h;              /**< Kernel height */
    int32_t kernel_w;              /**< Kernel width */
};

/** CSI-NN col2im params */
struct csinn_col2im_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t pad_h;                 /**< The number of vertical padding */
    int32_t pad_w;                 /**< The number of horizontal padding */
    int32_t stride_h;              /**< Vertical step */
    int32_t stride_w;              /**< Horizontal step */
};

/** CSI-NN reduce params */
struct csinn_reduce_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *out_strides;          /**< Outer step */
    int32_t *out_extents;          /**< Outer extension */
    int32_t n;                     /**< The number of latitudes on the outer layer */
    int32_t *inner_strides;        /**< Inner step */
    int32_t *inner_extents;        /**< Inner extension */
    int32_t m;                     /**< The number of latitudes in the inner layer */
    int32_t *axis;                 /**< Axis */
    int32_t axis_count;            /**< The number of axes */
    bool keepdims;                 /**< The number of latitude saved */
};

/** CSI-NN reorg params */
struct csinn_reorg_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t stride;                /**< The step size during reorg recombination */
};

/** CSI-NN segment params */
struct csinn_segment_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t num_segments;          /**< Number of segments */
    bool unsorted;                 /**< Indicates whether to sort */
};

/** CSI-NN cumsum params */
struct csinn_cumsum_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t axis;                  /**< Axis */
    bool exclusive;                /**< Exclusive */
};

/** CSI-NN cumprod params */
struct csinn_cumprod_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t axis;                  /**< Axis */
    bool exclusive;                /**< Exclusive */
};

/** CSI-NN broadcast_to params */
struct csinn_broadcast_to_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *shape;                /**< Output shape */
    int32_t shape_count;           /**< The number of output latitudes */
};

/** CSI-NN clip params */
struct csinn_clip_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    float min_value;               /**< Minimum value */
    float max_value;               /**< Maximum value */
};

/** CSI-NN stride_slice params */
struct csinn_strided_slice_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t *begin;                /**< Start position */
    int32_t *end;                  /**< End position */
    int32_t *stride;               /**< Step */
    int32_t slice_count;           /**< Number of slice */
};

/** CSI-NN shuffle_channel params */
struct csinn_shuffle_channel_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t group;                 /**< The number of groups */
};

/** CSI-NN topk params */
struct csinn_topk_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t k;                     /**< The parameter k of TOP-k */
};

/** CSI-NN non_max_suppression params */
struct csinn_non_max_suppression_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t max_output_size;       /**< The maximum number of outputs */
    float iou_threshold;           /**< iou threshold */
    // float score_threshold;
};

/** CSI-NN cast params */
struct csinn_cast_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    enum csinn_dtype_enum dtype;   /**< The destination type */
};

// modyfied to use asr model
struct csinn_layer_norm_params {
    struct csinn_params_base base;
    float epsilon;
    bool center;
    bool scale;
    int32_t axis;
};

struct csinn_asr_buffer_t {
    size_t writer_index;
    size_t buffer_lenth;  // lenth of buffer
    size_t data_lenth;    // lenth of data
    uint8_t *buffer;
    uint8_t flag;
};

struct csinn_cache_matmul_params {
    struct csinn_params_base base;
    struct csinn_asr_buffer_t asr_buffer;
    int32_t *cache_shape;
    int32_t *shape;
    int32_t *axes;
    void *data;
};

struct csinn_cache_conv1d_params {
    struct csinn_params_base base;
    struct csinn_asr_buffer_t asr_buffer;
    int32_t *cache_shape;
    int32_t *in_shape;
    int32_t group;
    int32_t stride_width;
    int32_t dilation_width;
    int32_t pad_left;
    int32_t pad_right;
    void *data;
};

/** CSI-NN conv1d params */
struct csinn_conv1d_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    int32_t group;                 /**< The number of convolutional groups */
    int32_t stride_width;          /**< Horizontal step */
    int32_t dilation_width;        /**< Horizontal expansion coefficient */
    int32_t pad_left;              /**< The number of left padding */
    int32_t pad_right;             /**< The number of right padding */
};

/** CSI-NN instance normalization params */
struct csinn_instance_norm_params {
    struct csinn_params_base base; /**< The basic information of the operator */
    float epsilon;
};

/**
 * @}
 */

#endif  // INCLUDE_CSI_INTERNAL_H_
