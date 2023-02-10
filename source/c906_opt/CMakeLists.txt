
if(CONFIG_C906_SOURCE)
    list(APPEND C906_SRCS source/c906_opt/utils.c)
    list(APPEND C906_SRCS source/c906_opt/setup.c)
    list(APPEND C906_SRCS source/c906_opt/hpm.c)
    list(APPEND C906_SRCS source/c906_opt/shl_c906_i8_to_f32.S)
    list(APPEND C906_SRCS source/c906_opt/shl_c906_u8_to_f32.S)
    list(APPEND C906_SRCS source/c906_opt/shl_c906_f32_to_i8.S)
    list(APPEND C906_SRCS source/c906_opt/shl_c906_f32_to_u8.S)
    list(APPEND C906_SRCS source/c906_opt/capability.c)
endif()

if(CONFIG_C906_ABS_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/abs.c)
endif()

if(CONFIG_C906_ABS_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/abs.c)
endif()

if(CONFIG_C906_ADD_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/add.c)
endif()

if(CONFIG_C906_ADD_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/add.c)
endif()

if(CONFIG_C906_AVERAGEPOOL_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/avgpool.c)
endif()

if(CONFIG_C906_AVERAGEPOOL_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/avgpool.c)
endif()

if(CONFIG_C906_CLIP_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/clip.c)
endif()

if(CONFIG_C906_CLIP_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/clip.c)
endif()

if(CONFIG_C906_CACHE_CONV1D_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/cache_conv1d.c)
endif()

if(CONFIG_C906_CACHE_MATMUL_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/cache_matmul.c)
endif()

if(CONFIG_C906_CONCAT_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/concat.c)
endif()

if(CONFIG_C906_CONCAT_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/concat.c)
endif()

if(CONFIG_C906_CONVOLUTION_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/convolution_1x1_fp32.c)
    list(APPEND C906_SRCS source/c906_opt/fp32/convolution_3x3_fp32.c)
    list(APPEND C906_SRCS source/c906_opt/fp32/convolution_sgemm_fp32.c)
    list(APPEND C906_SRCS source/c906_opt/fp32/convolution.c)
endif()

if(CONFIG_C906_CONVOLUTION_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/convolution_1x1_fp16.c)
    list(APPEND C906_SRCS source/c906_opt/fp16/convolution_3x3_fp16.c)
    list(APPEND C906_SRCS source/c906_opt/fp16/convolution_gemm_fp16.c)
    list(APPEND C906_SRCS source/c906_opt/fp16/convolution.c)
endif()

if(CONFIG_C906_CONVOLUTION1D_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/convolution1d.c)
endif()

if(CONFIG_C906_CONVOLUTION1D_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/convolution1d.c)
endif()

if(CONFIG_C906_DEPTHWISE_CONVOLUTION_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/depthwise_convolution_3x3_fp32.c)
    list(APPEND C906_SRCS source/c906_opt/fp32/depthwise_convolution_3x3_pack4_fp32.c)
    list(APPEND C906_SRCS source/c906_opt/fp32/depthwise_convolution_5x5_fp32.c)
    list(APPEND C906_SRCS source/c906_opt/fp32/depthwise_convolution_relu_3x3_fp32.c)
    list(APPEND C906_SRCS source/c906_opt/fp32/depthwise_convolution_relu_3x3_pack4_fp32.c)
    list(APPEND C906_SRCS source/c906_opt/fp32/depthwise_convolution_relu_5x5_fp32.c)
    list(APPEND C906_SRCS source/c906_opt/fp32/depthwise_convolution.c)
endif()

if(CONFIG_C906_DEPTHWISE_CONVOLUTION_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/depthwise_convolution_3x3_fp16.c)
    list(APPEND C906_SRCS source/c906_opt/fp16/depthwise_convolution_3x3_pack8_fp16.c)
    list(APPEND C906_SRCS source/c906_opt/fp16/depthwise_convolution_fp16.c)
    list(APPEND C906_SRCS source/c906_opt/fp16/depthwise_convolution.c)
endif()

if(CONFIG_C906_DEPTHWISE_CONVOLUTION1D_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/depthwise_convolution1d.c)
endif()

if(CONFIG_C906_DIV_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/div.c)
endif()

if(CONFIG_C906_DIV_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/div.c)
endif()

if(CONFIG_C906_FULLYCONNECTED_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/fullyconnected.c)
endif()

if(CONFIG_C906_GEMM_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/gemm_fp32.c)
endif()

if(CONFIG_C906_GEMM_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/gemm_fp16.c)
endif()

if(CONFIG_C906_GEMV_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/gemv_fp16.c)
endif()

if(CONFIG_C906_GLOBAL_AVERAGEPOOL_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/global_avgpool.c)
endif()

if(CONFIG_C906_GLOBAL_AVERAGEPOOL_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/global_avgpool.c)
endif()

if(CONFIG_C906_GLOBAL_MAXPOOL_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/global_maxpool.c)
endif()

if(CONFIG_C906_GLOBAL_MAXPOOL_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/global_maxpool.c)
endif()

if(CONFIG_C906_LEAKY_RELU_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/leaky_relu.c)
endif()

if(CONFIG_C906_LEAKY_RELU_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/leaky_relu.c)
endif()

if(CONFIG_C906_LRN_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/lrn.c)
endif()

if(CONFIG_C906_MATMUL_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/matmul.c)
endif()

if(CONFIG_C906_MAXPOOL_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/maxpool.c)
endif()

if(CONFIG_C906_MAXPOOL_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/maxpool.c)
endif()

if(CONFIG_C906_MUL_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/mul.c)
endif()

if(CONFIG_C906_MUL_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/mul.c)
endif()

if(CONFIG_C906_MINIMUM_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/minimum.c)
endif()

if(CONFIG_C906_MINIMUM_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/minimum.c)
endif()

if(CONFIG_C906_PAD_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/pad.c)
endif()

if(CONFIG_C906_PAD_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/pad.c)
endif()

if(CONFIG_C906_PRELU_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/prelu.c)
endif()

if(CONFIG_C906_PRELU_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/prelu.c)
endif()

if(CONFIG_C906_RELU_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/relu.c)
endif()

if(CONFIG_C906_RELU_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/relu.c)
endif()

if(CONFIG_C906_RELU1_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/relu1.c)
endif()

if(CONFIG_C906_RELU1_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/relu1.c)
endif()

if(CONFIG_C906_RELU6_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/relu6.c)
endif()

if(CONFIG_C906_RELU6_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/relu6.c)
endif()

if(CONFIG_C906_RESHAPE_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/reshape.c)
endif()

if(CONFIG_C906_SPLIT_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/split.c)
endif()

if(CONFIG_C906_SPLIT_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/split.c)
endif()

if(CONFIG_C906_SUB_FP32)
    list(APPEND C906_SRCS source/c906_opt/fp32/sub.c)
endif()

if(CONFIG_C906_SUB_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/sub.c)
endif()

if(CONFIG_C906_REDUCE_SUM_FP16)
    list(APPEND C906_SRCS source/c906_opt/fp16/reduce_sum.c)
endif()
