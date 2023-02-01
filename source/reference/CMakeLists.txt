
if(CONFIG_C_REFERENCE_SOURCE)
    list(APPEND REF_SRCS source/reference/utils.c)
    list(APPEND REF_SRCS source/reference/setup.c)
endif()

if(CONFIG_C_REFERENCE_ABS)
    list(APPEND REF_SRCS source/reference/abs.c)
endif()

if(CONFIG_C_REFERENCE_ACOS)
    list(APPEND REF_SRCS source/reference/acos.c)
endif()

if(CONFIG_C_REFERENCE_ACOSH)
    list(APPEND REF_SRCS source/reference/acosh.c)
endif()

if(CONFIG_C_REFERENCE_ADD)
    list(APPEND REF_SRCS source/reference/add.c)
endif()

if(CONFIG_C_REFERENCE_AND)
    list(APPEND REF_SRCS source/reference/and.c)
endif()

if(CONFIG_C_REFERENCE_ARANGE)
    list(APPEND REF_SRCS source/reference/arange.c)
endif()

if(CONFIG_C_REFERENCE_ARGMAX)
    list(APPEND REF_SRCS source/reference/argmax.c)
endif()

if(CONFIG_C_REFERENCE_ARGMIN)
    list(APPEND REF_SRCS source/reference/argmin.c)
endif()

if(CONFIG_C_REFERENCE_ASIN)
    list(APPEND REF_SRCS source/reference/asin.c)
endif()

if(CONFIG_C_REFERENCE_ASINH)
    list(APPEND REF_SRCS source/reference/asinh.c)
endif()

if(CONFIG_C_REFERENCE_ATAN)
    list(APPEND REF_SRCS source/reference/atan.c)
endif()

if(CONFIG_C_REFERENCE_ATANH)
    list(APPEND REF_SRCS source/reference/atanh.c)
endif()

if(CONFIG_C_REFERENCE_AVERAGEPOOL)
    list(APPEND REF_SRCS source/reference/averagepool.c)
endif()

if(CONFIG_C_REFERENCE_AVERAGEPOOL3D)
    list(APPEND REF_SRCS source/reference/averagepool3d.c)
endif()

if(CONFIG_C_REFERENCE_BATCH_NORMALIZATION)
    list(APPEND REF_SRCS source/reference/batch_normalization.c)
endif()

if(CONFIG_C_REFERENCE_BATCH_TO_SPACE)
    list(APPEND REF_SRCS source/reference/batch_to_space.c)
endif()

if(CONFIG_C_REFERENCE_BROADCAST_TO)
    list(APPEND REF_SRCS source/reference/broadcast_to.c)
endif()

if(CONFIG_C_REFERENCE_CACHE_CONV1D)
    list(APPEND REF_SRCS source/reference/cache_conv1d.c)
endif()

if(CONFIG_C_REFERENCE_CACHE_MATMUL)
    list(APPEND REF_SRCS source/reference/cache_matmul.c)
endif()

if(CONFIG_C_REFERENCE_CEIL)
    list(APPEND REF_SRCS source/reference/ceil.c)
endif()

if(CONFIG_C_REFERENCE_COL2IM)
    list(APPEND REF_SRCS source/reference/col2im.c)
endif()

if(CONFIG_C_REFERENCE_CLIP)
    list(APPEND REF_SRCS source/reference/clip.c)
endif()

if(CONFIG_C_REFERENCE_CONCAT)
    list(APPEND REF_SRCS source/reference/concat.c)
endif()

if(CONFIG_C_REFERENCE_CONVOLUTION_CHANNEL)
    list(APPEND REF_SRCS source/reference/convolution_channel.c)
endif()

if(CONFIG_C_REFERENCE_CONVOLUTION_RELU)
    list(APPEND REF_SRCS source/reference/convolution_relu.c)
endif()

if(CONFIG_C_REFERENCE_CONVOLUTION_RELU6)
    list(APPEND REF_SRCS source/reference/convolution_relu6.c)
endif()

if(CONFIG_C_REFERENCE_CONVOLUTION)
    list(APPEND REF_SRCS source/reference/convolution.c)
endif()

if(CONFIG_C_REFERENCE_CONVOLUTION1D)
    list(APPEND REF_SRCS source/reference/convolution1d.c)
endif()

if(CONFIG_C_REFERENCE_CONVOLUTION3D)
    list(APPEND REF_SRCS source/reference/convolution3d.c)
endif()

if(CONFIG_C_REFERENCE_COS)
    list(APPEND REF_SRCS source/reference/cos.c)
endif()

if(CONFIG_C_REFERENCE_COSH)
    list(APPEND REF_SRCS source/reference/cosh.c)
endif()

if(CONFIG_C_REFERENCE_CUMPROD)
    list(APPEND REF_SRCS source/reference/cumprod.c)
endif()

if(CONFIG_C_REFERENCE_CUMSUM)
    list(APPEND REF_SRCS source/reference/cumsum.c)
endif()

if(CONFIG_C_REFERENCE_DATA_CONVERT)
    list(APPEND REF_SRCS source/reference/data_convert.c)
endif()

if(CONFIG_C_REFERENCE_DECONVOLUTION)
    list(APPEND REF_SRCS source/reference/deconvolution.c)
endif()

if(CONFIG_C_REFERENCE_DECONVOLUTION3D)
    list(APPEND REF_SRCS source/reference/deconvolution3d.c)
endif()

if(CONFIG_C_REFERENCE_DEPTH_TO_SPACE)
    list(APPEND REF_SRCS source/reference/depth_to_space.c)
endif()

if(CONFIG_C_REFERENCE_DIV)
    list(APPEND REF_SRCS source/reference/div.c)
endif()

if(CONFIG_C_REFERENCE_ELU)
    list(APPEND REF_SRCS source/reference/elu.c)
endif()

if(CONFIG_C_REFERENCE_EQUAL)
    list(APPEND REF_SRCS source/reference/equal.c)
endif()

if(CONFIG_C_REFERENCE_ERF)
    list(APPEND REF_SRCS source/reference/erf.c)
endif()

if(CONFIG_C_REFERENCE_EXP)
    list(APPEND REF_SRCS source/reference/exp.c)
endif()

if(CONFIG_C_REFERENCE_EXPAND_DIMS)
    list(APPEND REF_SRCS source/reference/expand_dims.c)
endif()

if(CONFIG_C_REFERENCE_EXPM1)
    list(APPEND REF_SRCS source/reference/expm1.c)
endif()

if(CONFIG_C_REFERENCE_FLATTEN)
    list(APPEND REF_SRCS source/reference/flatten.c)
endif()

if(CONFIG_C_REFERENCE_FLOOR_DIVIDE)
    list(APPEND REF_SRCS source/reference/floor_divide.c)
endif()

if(CONFIG_C_REFERENCE_FLOOR_MOD)
    list(APPEND REF_SRCS source/reference/floor_mod.c)
endif()

if(CONFIG_C_REFERENCE_FLOOR)
    list(APPEND REF_SRCS source/reference/floor.c)
endif()

if(CONFIG_C_REFERENCE_FSMN)
    list(APPEND REF_SRCS source/reference/fsmn.c)
endif()

if(CONFIG_C_REFERENCE_FULLYCONNECTED)
    list(APPEND REF_SRCS source/reference/fullyconnected.c)
endif()

if(CONFIG_C_REFERENCE_GATHER_ND)
    list(APPEND REF_SRCS source/reference/gather_nd.c)
endif()

if(CONFIG_C_REFERENCE_GATHER)
    list(APPEND REF_SRCS source/reference/gather.c)
endif()

if(CONFIG_C_REFERENCE_GLOBAL_AVERAGEPOOL)
    list(APPEND REF_SRCS source/reference/global_averagepool.c)
endif()

if(CONFIG_C_REFERENCE_GLOBAL_MAXPOOL)
    list(APPEND REF_SRCS source/reference/global_maxpool.c)
endif()

if(CONFIG_C_REFERENCE_GREATER_EQUAL)
    list(APPEND REF_SRCS source/reference/greater_equal.c)
endif()

if(CONFIG_C_REFERENCE_GREATER)
    list(APPEND REF_SRCS source/reference/greater.c)
endif()

if(CONFIG_C_REFERENCE_HARD_SIGMOID)
    list(APPEND REF_SRCS source/reference/hard_sigmoid.c)
endif()

if(CONFIG_C_REFERENCE_IM2COL)
    list(APPEND REF_SRCS source/reference/im2col.c)
endif()

if(CONFIG_C_REFERENCE_ISNAN)
    list(APPEND REF_SRCS source/reference/isnan.c)
endif()

if(CONFIG_C_REFERENCE_L2_NORMALIZATION)
    list(APPEND REF_SRCS source/reference/l2_normalization.c)
endif()

if(CONFIG_C_REFERENCE_L2POOL)
    list(APPEND REF_SRCS source/reference/l2pool.c)
endif()

if(CONFIG_C_REFERENCE_LAYER_NORMAL)
    list(APPEND REF_SRCS source/reference/layer_normal.c)
endif()

if(CONFIG_C_REFERENCE_LEAKY_RELU)
    list(APPEND REF_SRCS source/reference/leaky_relu.c)
endif()

if(CONFIG_C_REFERENCE_LESS_EQUAL)
    list(APPEND REF_SRCS source/reference/less_equal.c)
endif()

if(CONFIG_C_REFERENCE_LESS)
    list(APPEND REF_SRCS source/reference/less.c)
endif()

if(CONFIG_C_REFERENCE_LOG_SOFTMAX)
    list(APPEND REF_SRCS source/reference/log_softmax.c)
endif()

if(CONFIG_C_REFERENCE_LOG)
    list(APPEND REF_SRCS source/reference/log.c)
endif()

if(CONFIG_C_REFERENCE_LOG1P)
    list(APPEND REF_SRCS source/reference/log1p.c)
endif()

if(CONFIG_C_REFERENCE_LOGICAL_AND)
    list(APPEND REF_SRCS source/reference/logical_and.c)
endif()

if(CONFIG_C_REFERENCE_LOGICAL_NOT)
    list(APPEND REF_SRCS source/reference/logical_not.c)
endif()

if(CONFIG_C_REFERENCE_LOGICAL_OR)
    list(APPEND REF_SRCS source/reference/logical_or.c)
endif()

if(CONFIG_C_REFERENCE_LOGICAL_XOR)
    list(APPEND REF_SRCS source/reference/logical_xor.c)
endif()

if(CONFIG_C_REFERENCE_LRN)
    list(APPEND REF_SRCS source/reference/lrn.c)
endif()

if(CONFIG_C_REFERENCE_MATMUL)
    list(APPEND REF_SRCS source/reference/matmul.c)
endif()

if(CONFIG_C_REFERENCE_MAX)
    list(APPEND REF_SRCS source/reference/max.c)
endif()

if(CONFIG_C_REFERENCE_MAXIMUM)
    list(APPEND REF_SRCS source/reference/maximum.c)
endif()

if(CONFIG_C_REFERENCE_MAXPOOL)
    list(APPEND REF_SRCS source/reference/maxpool.c)
endif()

if(CONFIG_C_REFERENCE_MAXPOOL2D_LOCAT)
    list(APPEND REF_SRCS source/reference/maxpool2d_locat.c)
endif()

if(CONFIG_C_REFERENCE_MAXPOOL3D)
    list(APPEND REF_SRCS source/reference/maxpool3d.c)
endif()

if(CONFIG_C_REFERENCE_MEAN)
    list(APPEND REF_SRCS source/reference/mean.c)
endif()

if(CONFIG_C_REFERENCE_MIN)
    list(APPEND REF_SRCS source/reference/min.c)
endif()

if(CONFIG_C_REFERENCE_MINIMUM)
    list(APPEND REF_SRCS source/reference/minimum.c)
endif()

if(CONFIG_C_REFERENCE_MOD)
    list(APPEND REF_SRCS source/reference/mod.c)
endif()

if(CONFIG_C_REFERENCE_MUL)
    list(APPEND REF_SRCS source/reference/mul.c)
endif()

if(CONFIG_C_REFERENCE_NDARRAY_SIZE)
    list(APPEND REF_SRCS source/reference/ndarray_size.c)
endif()

if(CONFIG_C_REFERENCE_NEGATIVE)
    list(APPEND REF_SRCS source/reference/negative.c)
endif()

if(CONFIG_C_REFERENCE_NON_MAX_SUPPRESSION)
    list(APPEND REF_SRCS source/reference/non_max_suppression.c)
endif()

if(CONFIG_C_REFERENCE_NOT_EQUAL)
    list(APPEND REF_SRCS source/reference/not_equal.c)
endif()

if(CONFIG_C_REFERENCE_NOT)
    list(APPEND REF_SRCS source/reference/not.c)
endif()

if(CONFIG_C_REFERENCE_OR)
    list(APPEND REF_SRCS source/reference/or.c)
endif()

if(CONFIG_C_REFERENCE_PAD)
    list(APPEND REF_SRCS source/reference/pad.c)
endif()

if(CONFIG_C_REFERENCE_POWER)
    list(APPEND REF_SRCS source/reference/power.c)
endif()

if(CONFIG_C_REFERENCE_PRELU)
    list(APPEND REF_SRCS source/reference/prelu.c)
endif()

if(CONFIG_C_REFERENCE_PROD)
    list(APPEND REF_SRCS source/reference/prod.c)
endif()

if(CONFIG_C_REFERENCE_PROPOSAL)
    list(APPEND REF_SRCS source/reference/proposal.c)
endif()

if(CONFIG_C_REFERENCE_PSROIPOOLING)
    list(APPEND REF_SRCS source/reference/psroipooling.c)
endif()

if(CONFIG_C_REFERENCE_REDUCE_LOGSUMEXP)
    list(APPEND REF_SRCS source/reference/reduce_logsumexp.c)
endif()

if(CONFIG_C_REFERENCE_REDUCE_MAX)
    list(APPEND REF_SRCS source/reference/reduce_max.c)
endif()

if(CONFIG_C_REFERENCE_REDUCE_MEAN)
    list(APPEND REF_SRCS source/reference/reduce_mean.c)
endif()

if(CONFIG_C_REFERENCE_REDUCE_MIN)
    list(APPEND REF_SRCS source/reference/reduce_min.c)
endif()

if(CONFIG_C_REFERENCE_REDUCE_PROD)
    list(APPEND REF_SRCS source/reference/reduce_prod.c)
endif()

if(CONFIG_C_REFERENCE_REDUCE_SUM)
    list(APPEND REF_SRCS source/reference/reduce_sum.c)
endif()

if(CONFIG_C_REFERENCE_RELU)
    list(APPEND REF_SRCS source/reference/relu.c)
endif()

if(CONFIG_C_REFERENCE_RELU1)
    list(APPEND REF_SRCS source/reference/relu1.c)
endif()

if(CONFIG_C_REFERENCE_RELU6)
    list(APPEND REF_SRCS source/reference/relu6.c)
endif()

if(CONFIG_C_REFERENCE_RELUN)
    list(APPEND REF_SRCS source/reference/relun.c)
endif()

if(CONFIG_C_REFERENCE_RESHAPE)
    list(APPEND REF_SRCS source/reference/reshape.c)
endif()

if(CONFIG_C_REFERENCE_RESIZE)
    list(APPEND REF_SRCS source/reference/resize.c)
endif()

if(CONFIG_C_REFERENCE_REVERSE)
    list(APPEND REF_SRCS source/reference/reverse.c)
endif()

if(CONFIG_C_REFERENCE_ROIALIGN)
    list(APPEND REF_SRCS source/reference/roialign.c)
endif()

if(CONFIG_C_REFERENCE_ROIPOOL)
    list(APPEND REF_SRCS source/reference/roipool.c)
endif()

if(CONFIG_C_REFERENCE_ROUND)
    list(APPEND REF_SRCS source/reference/round.c)
endif()

if(CONFIG_C_REFERENCE_RSQRT)
    list(APPEND REF_SRCS source/reference/rsqrt.c)
endif()

if(CONFIG_C_REFERENCE_SCATTER)
    list(APPEND REF_SRCS source/reference/scatter.c)
endif()

if(CONFIG_C_REFERENCE_SEGMENT_MAX)
    list(APPEND REF_SRCS source/reference/segment_max.c)
endif()

if(CONFIG_C_REFERENCE_SEGMENT_MEAN)
    list(APPEND REF_SRCS source/reference/segment_mean.c)
endif()

if(CONFIG_C_REFERENCE_SEGMENT_MIN)
    list(APPEND REF_SRCS source/reference/segment_min.c)
endif()

if(CONFIG_C_REFERENCE_SEGMENT_PROD)
    list(APPEND REF_SRCS source/reference/segment_prod.c)
endif()

if(CONFIG_C_REFERENCE_SEGMENT_SUM)
    list(APPEND REF_SRCS source/reference/segment_sum.c)
endif()

if(CONFIG_C_REFERENCE_SELECT)
    list(APPEND REF_SRCS source/reference/select.c)
endif()

if(CONFIG_C_REFERENCE_SHAPE)
    list(APPEND REF_SRCS source/reference/shape.c)
endif()

if(CONFIG_C_REFERENCE_SHUFFLE_CHANNEL)
    list(APPEND REF_SRCS source/reference/shuffle_channel.c)
endif()

if(CONFIG_C_REFERENCE_SIGMOID)
    list(APPEND REF_SRCS source/reference/sigmoid.c)
endif()

if(CONFIG_C_REFERENCE_SIGN)
    list(APPEND REF_SRCS source/reference/sign.c)
endif()

if(CONFIG_C_REFERENCE_SIN)
    list(APPEND REF_SRCS source/reference/sin.c)
endif()

if(CONFIG_C_REFERENCE_SINH)
    list(APPEND REF_SRCS source/reference/sinh.c)
endif()

if(CONFIG_C_REFERENCE_SLICE)
    list(APPEND REF_SRCS source/reference/slice.c)
endif()

if(CONFIG_C_REFERENCE_SOFTMAX)
    list(APPEND REF_SRCS source/reference/softmax.c)
endif()

if(CONFIG_C_REFERENCE_SOFTPLUS)
    list(APPEND REF_SRCS source/reference/softplus.c)
endif()

if(CONFIG_C_REFERENCE_SOFTRELU)
    list(APPEND REF_SRCS source/reference/softrelu.c)
endif()

if(CONFIG_C_REFERENCE_SOFTSIGN)
    list(APPEND REF_SRCS source/reference/softsign.c)
endif()

if(CONFIG_C_REFERENCE_SPACE_TO_BATCH)
    list(APPEND REF_SRCS source/reference/space_to_batch.c)
endif()

if(CONFIG_C_REFERENCE_SPACE_TO_DEPTH)
    list(APPEND REF_SRCS source/reference/space_to_depth.c)
endif()

if(CONFIG_C_REFERENCE_SPLIT)
    list(APPEND REF_SRCS source/reference/split.c)
endif()

if(CONFIG_C_REFERENCE_SQRT)
    list(APPEND REF_SRCS source/reference/sqrt.c)
endif()

if(CONFIG_C_REFERENCE_SQUARE)
    list(APPEND REF_SRCS source/reference/square.c)
endif()

if(CONFIG_C_REFERENCE_SQUEEZE)
    list(APPEND REF_SRCS source/reference/squeeze.c)
endif()

if(CONFIG_C_REFERENCE_STACK)
    list(APPEND REF_SRCS source/reference/stack.c)
endif()

if(CONFIG_C_REFERENCE_STRIDED_SLICE)
    list(APPEND REF_SRCS source/reference/strided_slice.c)
endif()

if(CONFIG_C_REFERENCE_SUB)
    list(APPEND REF_SRCS source/reference/sub.c)
endif()

if(CONFIG_C_REFERENCE_SUM)
    list(APPEND REF_SRCS source/reference/sum.c)
endif()

if(CONFIG_C_REFERENCE_TAN)
    list(APPEND REF_SRCS source/reference/tan.c)
endif()

if(CONFIG_C_REFERENCE_TANH)
    list(APPEND REF_SRCS source/reference/tanh.c)
endif()

if(CONFIG_C_REFERENCE_THRESHOLD_RELU)
    list(APPEND REF_SRCS source/reference/threshold_relu.c)
endif()

if(CONFIG_C_REFERENCE_TILE)
    list(APPEND REF_SRCS source/reference/tile.c)
endif()

if(CONFIG_C_REFERENCE_TOPK)
    list(APPEND REF_SRCS source/reference/topk.c)
endif()

if(CONFIG_C_REFERENCE_TRANSPOSE)
    list(APPEND REF_SRCS source/reference/transpose.c)
endif()

if(CONFIG_C_REFERENCE_TRUNC)
    list(APPEND REF_SRCS source/reference/trunc.c)
endif()

if(CONFIG_C_REFERENCE_UNPOOLING)
    list(APPEND REF_SRCS source/reference/unpooling.c)
endif()

if(CONFIG_C_REFERENCE_UNSTACK)
    list(APPEND REF_SRCS source/reference/unstack.c)
endif()

if(CONFIG_C_REFERENCE_XOR)
    list(APPEND REF_SRCS source/reference/xor.c)
endif()

if(CONFIG_C_REFERENCE_YUV_RGB_SCALE)
    list(APPEND REF_SRCS source/reference/yuv_rgb_scale.c)
endif()

if(CONFIG_C_REFERENCE_ONE_HOT)
    list(APPEND REF_SRCS source/reference/one_hot.c)
endif()

if(CONFIG_C_REFERENCE_WHERE)
    list(APPEND REF_SRCS source/reference/where.c)
endif()
