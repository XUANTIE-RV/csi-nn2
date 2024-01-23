
if(CONFIG_THEAD_RVV_SOURCE)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/utils.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/setup.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/reorder.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/data_convert.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/capability.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/binary_broadcast.c)
endif()

if(CONFIG_THEAD_RVV_ADD_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/add.c)
endif()

if(CONFIG_THEAD_RVV_ADD_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/add.c)
endif()

if(CONFIG_THEAD_RVV_ADD_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/add.c)
endif()

if(CONFIG_THEAD_RVV_AVERAGEPOOL_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/avgpool_2x2_fp32.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/avgpool_3x3_fp32.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/avgpool_fp32_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/avgpool_fp32_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/avgpool_fp32_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/avgpool.c)
endif()

if(CONFIG_THEAD_RVV_AVERAGEPOOL_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/avgpool_2x2_fp16.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/avgpool_3x3_fp16.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/avgpool_fp16_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/avgpool_fp16_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/avgpool.c)
endif()

if(CONFIG_THEAD_RVV_AVERAGEPOOL_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/avgpool_int8_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/avgpool_int8_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/avgpool.c)
endif()

if(CONFIG_THEAD_RVV_CLIP_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/clip.c)
endif()

if(CONFIG_THEAD_RVV_CLIP_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/clip.c)
endif()

if(CONFIG_THEAD_RVV_CLIP_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/clip.c)
endif()

if(CONFIG_THEAD_RVV_CONCAT_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/concat.c)
endif()

if(CONFIG_THEAD_RVV_CONCAT_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/concat.c)
endif()

if(CONFIG_THEAD_RVV_CONCAT_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/concat.c)
endif()

if(CONFIG_THEAD_RVV_CONVOLUTION_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution_1x1_fp32_pack1ton.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution_1x1_fp32_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution_1x1_fp32_packnto1.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution_1x1_fp32.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution_3x3_fp32_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution_gemm_fp32_pack1ton.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution_gemm_fp32_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution_gemm_fp32_packnto1.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution_gemm_fp32.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution.c)
endif()

if(CONFIG_THEAD_RVV_CONVOLUTION_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution_1x1_fp16_pack1ton.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution_1x1_fp16_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution_1x1_fp16_packnto1.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution_1x1_fp16.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution_3x3_fp16_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution_gemm_fp16_pack1ton.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution_gemm_fp16_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution_gemm_fp16_packnto1.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution_gemm_fp16.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution_direct_fp16.c)
endif()

if(CONFIG_THEAD_RVV_CONVOLUTION_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution_1x1_int8_pack1ton.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution_1x1_int8_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution_1x1_int8_packnto1.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution_1x1_int8.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution_3x3_int8_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution_gemm_int8_pack1ton.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution_gemm_int8_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution_gemm_int8_packnto1.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution_gemm_int8.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution.c)
endif()

if(CONFIG_THEAD_RVV_CONVOLUTION_INT4)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int4/convolution_1x1_int4_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int4/convolution_1x1_int4.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int4/convolution_gemm_int4_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int4/convolution_gemm_int4.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int4/convolution.c)
endif()

if(CONFIG_THEAD_RVV_CONVOLUTION1D_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution1d.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/convolution1d_gemm_fp32.c)
endif()

if(CONFIG_THEAD_RVV_CONVOLUTION1D_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution1d.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/convolution1d_gemm_fp16.c)
endif()

if(CONFIG_THEAD_RVV_CONVOLUTION1D_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution1d.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/convolution1d_1_int8.c)
endif()

if(CONFIG_THEAD_RVV_DEPTHWISE_CONVOLUTION_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/depthwise_convolution_3x3_fp32_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/depthwise_convolution_3x3_fp32.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/depthwise_convolution_fp32_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/depthwise_convolution_fp32_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/depthwise_convolution.c)
endif()

if(CONFIG_THEAD_RVV_DEPTHWISE_CONVOLUTION_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/depthwise_convolution_3x3_fp16_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/depthwise_convolution_3x3_fp16.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/depthwise_convolution_fp16_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/depthwise_convolution_fp16_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/depthwise_convolution.c)
endif()

if(CONFIG_THEAD_RVV_DEPTHWISE_CONVOLUTION_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/depthwise_convolution_3x3_int8_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/depthwise_convolution_3x3_int8.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/depthwise_convolution_int8_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/depthwise_convolution_int8_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/depthwise_convolution.c)
endif()

if(CONFIG_THEAD_RVV_DEPTHWISE_CONVOLUTION_INT4)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int4/depthwise_convolution_3x3_int4.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int4/depthwise_convolution.c)
endif()

if(CONFIG_THEAD_RVV_DEPTHWISE_CONVOLUTION1D_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/depthwise_convolution1d_int8.c)
endif()

if(CONFIG_THEAD_RVV_DECONVOLUTION_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/deconvolution_gemm_fp32.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/deconvolution.c)
endif()

if(CONFIG_THEAD_RVV_DECONVOLUTION_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/deconvolution_gemm_fp16.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/deconvolution.c)
endif()

if(CONFIG_THEAD_RVV_DIV_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/div.c)
endif()

if(CONFIG_THEAD_RVV_DIV_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/div.c)
endif()

if(CONFIG_THEAD_RVV_DIV_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/div.c)
endif()

if(CONFIG_THEAD_RVV_EMBEDDING_INT32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int32/embedding.c)
endif()

if(CONFIG_THEAD_RVV_ERF_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/erf.c)
endif()

if(CONFIG_THEAD_RVV_ERF_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/erf.c)
endif()

if(CONFIG_THEAD_RVV_ERF_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/erf.c)
endif()

if(CONFIG_THEAD_RVV_EXPAND_DIMS_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/expand_dims.c)
endif()

if(CONFIG_THEAD_RVV_EXPAND_DIMS_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/expand_dims.c)
endif()

if(CONFIG_THEAD_RVV_FULLYCONNECTED_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/gemm_fp32_a0b1.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/fullyconnected_fp32.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/fullyconnected.c)
endif()

if(CONFIG_THEAD_RVV_FULLYCONNECTED_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/gemm_fp16_a0b1.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/fullyconnected_fp16.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/fullyconnected.c)
endif()

if(CONFIG_THEAD_RVV_FULLYCONNECTED_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/gemm_int8_a0b1.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/fullyconnected_int8.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/fullyconnected.c)
endif()

if(CONFIG_THEAD_RVV_FULLYCONNECTED_INT4)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int4/fullyconnected_int4.c)
endif()

if(CONFIG_THEAD_RVV_GATHER_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/gather.c)
endif()

if(CONFIG_THEAD_RVV_GATHER_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/gather.c)
endif()

if(CONFIG_THEAD_RVV_GATHER_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/gather.c)
endif()

if(CONFIG_THEAD_RVV_GEMM_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/gemm_fp32_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/gemm_fp32.c)
endif()

if(CONFIG_THEAD_RVV_GEMM_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/gemm_fp16_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/gemm_fp16.c)
endif()

if(CONFIG_THEAD_RVV_GEMM_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/gemm_int8_dot_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/gemm_int8_dot.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/gemm_int8_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/gemm_int8.c)
endif()

if(CONFIG_THEAD_RVV_GEMM_INT4)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int4/gemm_int4_dot_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int4/gemm_int4_dot.c)
endif()

if(CONFIG_THEAD_RVV_GLOBAL_AVERAGEPOOL_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/global_avgpool_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/global_avgpool_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/global_avgpool.c)
endif()

if(CONFIG_THEAD_RVV_GLOBAL_AVERAGEPOOL_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/global_avgpool_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/global_avgpool_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/global_avgpool.c)
endif()

if(CONFIG_THEAD_RVV_GLOBAL_AVERAGEPOOL_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/global_avgpool_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/global_avgpool_packn.c)
endif()

if(CONFIG_THEAD_RVV_GLOBAL_MAXPOOL_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/global_maxpool_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/global_maxpool_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/global_maxpool.c)
endif()

if(CONFIG_THEAD_RVV_GLOBAL_MAXPOOL_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/global_maxpool_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/global_maxpool_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/global_maxpool.c)
endif()

if(CONFIG_THEAD_RVV_GLOBAL_MAXPOOL_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/global_maxpool_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/global_maxpool_packn.c)
endif()

if(CONFIG_THEAD_RVV_LAYER_NORM_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/layer_norm.c)
endif()

if(CONFIG_THEAD_RVV_LAYER_NORM_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/layer_norm.c)
endif()

if(CONFIG_THEAD_RVV_LAYER_NORM_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/layer_norm.c)
endif()

if(CONFIG_THEAD_RVV_LEAKY_RELU_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/leaky_relu.c)
endif()

if(CONFIG_THEAD_RVV_LEAKY_RELU_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/leaky_relu.c)
endif()

if(CONFIG_THEAD_RVV_LEAKY_RELU_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/leaky_relu.c)
endif()

if(CONFIG_THEAD_RVV_LLM_POS_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/llm_pos.c)
endif()

if(CONFIG_THEAD_RVV_MATMUL_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/matmul.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/gemm_fp32_block.c)
endif()

if(CONFIG_THEAD_RVV_MATMUL_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/matmul.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/gemm_fp16_block.c)
endif()

if(CONFIG_THEAD_RVV_MATMUL_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/matmul.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/matmul_int8.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/matmul_int8_dot.c)
endif()

if(CONFIG_THEAD_RVV_MAXPOOL_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/maxpool_2x2_fp32.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/maxpool_3x3_fp32.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/maxpool_fp32_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/maxpool_fp32_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/maxpool.c)
endif()

if(CONFIG_THEAD_RVV_MAXPOOL_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/maxpool_2x2_fp16.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/maxpool_3x3_fp16.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/maxpool_fp16_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/maxpool_fp16_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/maxpool.c)
endif()

if(CONFIG_THEAD_RVV_MAXPOOL_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/maxpool_2x2_int8.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/maxpool_3x3_int8.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/maxpool_int8_packn.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/maxpool_int8_nhwc.c)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/maxpool.c)
endif()

if(CONFIG_THEAD_RVV_MUL_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/mul.c)
endif()

if(CONFIG_THEAD_RVV_MUL_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/mul.c)
endif()

if(CONFIG_THEAD_RVV_MUL_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/mul.c)
endif()

if(CONFIG_THEAD_RVV_PAD_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/pad.c)
endif()

if(CONFIG_THEAD_RVV_PAD_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/pad.c)
endif()

if(CONFIG_THEAD_RVV_PAD_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/pad.c)
endif()

if(CONFIG_THEAD_RVV_PRELU_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/prelu.c)
endif()

if(CONFIG_THEAD_RVV_PRELU_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/prelu.c)
endif()

if(CONFIG_THEAD_RVV_PRELU_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/prelu.c)
endif()

if(CONFIG_THEAD_RVV_REDUCE_SUM_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/reduce_sum.c)
endif()

if(CONFIG_THEAD_RVV_RELU_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/relu.c)
endif()

if(CONFIG_THEAD_RVV_RELU_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/relu.c)
endif()

if(CONFIG_THEAD_RVV_RELU_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/relu.c)
endif()

if(CONFIG_THEAD_RVV_RELU6_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/relu6.c)
endif()

if(CONFIG_THEAD_RVV_RELU6_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/relu6.c)
endif()

if(CONFIG_THEAD_RVV_RELU6_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/relu6.c)
endif()

if(CONFIG_THEAD_RVV_RESHAPE_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/reshape.c)
endif()

if(CONFIG_THEAD_RVV_RESHAPE_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/reshape.c)
endif()

if(CONFIG_THEAD_RVV_RESHAPE_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/reshape.c)
endif()

if(CONFIG_THEAD_RVV_RMS_NORM_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/rms_norm.c)
endif()

if(CONFIG_THEAD_RVV_RMS_NORM_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/rms_norm.c)
endif()

if(CONFIG_THEAD_RVV_RMS_NORM_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/rms_norm.c)
endif()

if(CONFIG_THEAD_RVV_ROPE_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/rope.c)
endif()

if(CONFIG_THEAD_RVV_ROPE_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/rope.c)
endif()

if(CONFIG_THEAD_RVV_SCALED_DOT_PRODUCT_ATTENTION_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/scaled_dot_product_attention.c)
endif()

if(CONFIG_THEAD_RVV_SCALED_DOT_PRODUCT_ATTENTION_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/scaled_dot_product_attention.c)
endif()

if(CONFIG_THEAD_RVV_SIGMOID_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/sigmoid.c)
endif()

if(CONFIG_THEAD_RVV_SIGMOID_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/sigmoid.c)
endif()

if(CONFIG_THEAD_RVV_SIGMOID_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/sigmoid.c)
endif()

if(CONFIG_THEAD_RVV_SILU_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/silu.c)
endif()

if(CONFIG_THEAD_RVV_SILU_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/silu.c)
endif()

if(CONFIG_THEAD_RVV_SILU_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/silu.c)
endif()

if(CONFIG_THEAD_RVV_SUB_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/sub.c)
endif()

if(CONFIG_THEAD_RVV_SUB_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/sub.c)
endif()

if(CONFIG_THEAD_RVV_SUB_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/sub.c)
endif()

if(CONFIG_THEAD_RVV_SOFTMAX_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/softmax.c)
endif()

if(CONFIG_THEAD_RVV_SOFTMAX_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/softmax.c)
endif()

if(CONFIG_THEAD_RVV_SOFTMAX_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/softmax.c)
endif()

if(CONFIG_THEAD_RVV_SPLIT_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/split.c)
endif()

if(CONFIG_THEAD_RVV_SPLIT_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/split.c)
endif()

if(CONFIG_THEAD_RVV_SPLIT_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/split.c)
endif()

if(CONFIG_THEAD_RVV_STRIDED_SLICE_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/strided_slice.c)
endif()

if(CONFIG_THEAD_RVV_TRANSPOSE_FP32)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp32/transpose.c)
endif()

if(CONFIG_THEAD_RVV_TRANSPOSE_FP16)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/fp16/transpose.c)
endif()

if(CONFIG_THEAD_RVV_TRANSPOSE_INT8)
    list(APPEND THEAD_RVV_SRCS_MOD source/thead_rvv/int8/transpose.c)
endif()
