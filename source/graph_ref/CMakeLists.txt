
if(CONFIG_GRAPH_REFERENCE_SOURCE)
    list(APPEND GREF_SRCS source/graph_ref/utils.c)
    list(APPEND GREF_SRCS source/graph_ref/setup.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SUBGRAPH)
    list(APPEND GREF_SRCS source/graph_ref/subgraph.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ABS)
    list(APPEND GREF_SRCS source/graph_ref/abs.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ACOS)
    list(APPEND GREF_SRCS source/graph_ref/acos.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ACOSH)
    list(APPEND GREF_SRCS source/graph_ref/acosh.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ADD)
    list(APPEND GREF_SRCS source/graph_ref/add.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ALL)
    list(APPEND GREF_SRCS source/graph_ref/all.c)
endif()

if(CONFIG_GRAPH_REFERENCE_AND)
    list(APPEND GREF_SRCS source/graph_ref/and.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ANY)
    list(APPEND GREF_SRCS source/graph_ref/any.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ARANGE)
    list(APPEND GREF_SRCS source/graph_ref/arange.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ARGMAX)
    list(APPEND GREF_SRCS source/graph_ref/argmax.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ARGMIN)
    list(APPEND GREF_SRCS source/graph_ref/argmin.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ASIN)
    list(APPEND GREF_SRCS source/graph_ref/asin.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ASINH)
    list(APPEND GREF_SRCS source/graph_ref/asinh.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ATAN)
    list(APPEND GREF_SRCS source/graph_ref/atan.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ATANH)
    list(APPEND GREF_SRCS source/graph_ref/atanh.c)
endif()

if(CONFIG_GRAPH_REFERENCE_AVERAGEPOOL)
    list(APPEND GREF_SRCS source/graph_ref/avgpool.c)
endif()

if(CONFIG_GRAPH_REFERENCE_AVERAGEPOOL3D)
    list(APPEND GREF_SRCS source/graph_ref/avgpool3d.c)
endif()

if(CONFIG_GRAPH_REFERENCE_BATCH_NORMALIZATION)
    list(APPEND GREF_SRCS source/graph_ref/bn.c)
endif()

if(CONFIG_GRAPH_REFERENCE_BATCH_TO_SPACE)
    list(APPEND GREF_SRCS source/graph_ref/batch_to_space.c)
endif()

if(CONFIG_GRAPH_REFERENCE_BATCH_TO_SPACE_ND)
    list(APPEND GREF_SRCS source/graph_ref/batch_to_space_nd.c)
endif()

if(CONFIG_GRAPH_REFERENCE_BROADCAST_TO)
    list(APPEND GREF_SRCS source/graph_ref/broadcast_to.c)
endif()

if(CONFIG_GRAPH_REFERENCE_CACHE_CONV1D)
    list(APPEND GREF_SRCS source/graph_ref/cache_conv1d.c)
endif()

if(CONFIG_GRAPH_REFERENCE_CACHE_MATMUL)
    list(APPEND GREF_SRCS source/graph_ref/cache_matmul.c)
endif()

if(CONFIG_GRAPH_REFERENCE_CEIL)
    list(APPEND GREF_SRCS source/graph_ref/ceil.c)
endif()

if(CONFIG_GRAPH_REFERENCE_CLIP)
    list(APPEND GREF_SRCS source/graph_ref/clip.c)
endif()

if(CONFIG_GRAPH_REFERENCE_COL2IM)
    list(APPEND GREF_SRCS source/graph_ref/col2im.c)
endif()

if(CONFIG_GRAPH_REFERENCE_CONCAT)
    list(APPEND GREF_SRCS source/graph_ref/concat.c)
endif()

if(CONFIG_GRAPH_REFERENCE_CONVOLUTION)
    list(APPEND GREF_SRCS source/graph_ref/convolution.c)
endif()

if(CONFIG_GRAPH_REFERENCE_CONVOLUTION1D)
    list(APPEND GREF_SRCS source/graph_ref/convolution1d.c)
endif()

if(CONFIG_GRAPH_REFERENCE_CONVOLUTION3D)
    list(APPEND GREF_SRCS source/graph_ref/convolution3d.c)
endif()

if(CONFIG_GRAPH_REFERENCE_COS)
    list(APPEND GREF_SRCS source/graph_ref/cos.c)
endif()

if(CONFIG_GRAPH_REFERENCE_COSH)
    list(APPEND GREF_SRCS source/graph_ref/cosh.c)
endif()

if(CONFIG_GRAPH_REFERENCE_CUMPROD)
    list(APPEND GREF_SRCS source/graph_ref/cumprod.c)
endif()

if(CONFIG_GRAPH_REFERENCE_CUMSUM)
    list(APPEND GREF_SRCS source/graph_ref/cumsum.c)
endif()

if(CONFIG_GRAPH_REFERENCE_DATA_CONVERT)
    list(APPEND GREF_SRCS source/graph_ref/data_convert.c)
endif()

if(CONFIG_GRAPH_REFERENCE_DECONVOLUTION)
    list(APPEND GREF_SRCS source/graph_ref/deconvolution.c)
endif()

if(CONFIG_GRAPH_REFERENCE_DECONVOLUTION3D)
    list(APPEND GREF_SRCS source/graph_ref/deconvolution3d.c)
endif()

if(CONFIG_GRAPH_REFERENCE_DEPTH_TO_SPACE)
    list(APPEND GREF_SRCS source/graph_ref/depth_to_space.c)
endif()

if(CONFIG_GRAPH_REFERENCE_DIV)
    list(APPEND GREF_SRCS source/graph_ref/div.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ELU)
    list(APPEND GREF_SRCS source/graph_ref/elu.c)
endif()

if(CONFIG_GRAPH_REFERENCE_EQUAL)
    list(APPEND GREF_SRCS source/graph_ref/equal.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ERF)
    list(APPEND GREF_SRCS source/graph_ref/erf.c)
endif()

if(CONFIG_GRAPH_REFERENCE_EXP)
    list(APPEND GREF_SRCS source/graph_ref/exp.c)
endif()

if(CONFIG_GRAPH_REFERENCE_EXPAND_DIMS)
    list(APPEND GREF_SRCS source/graph_ref/expand_dims.c)
endif()

if(CONFIG_GRAPH_REFERENCE_EXPM1)
    list(APPEND GREF_SRCS source/graph_ref/expm1.c)
endif()

if(CONFIG_GRAPH_REFERENCE_FLATTEN)
    list(APPEND GREF_SRCS source/graph_ref/flatten.c)
endif()

if(CONFIG_GRAPH_REFERENCE_FLOOR_DIVIDE)
    list(APPEND GREF_SRCS source/graph_ref/floor_divide.c)
endif()

if(CONFIG_GRAPH_REFERENCE_FLOOR_MOD)
    list(APPEND GREF_SRCS source/graph_ref/floor_mod.c)
endif()

if(CONFIG_GRAPH_REFERENCE_FLOOR)
    list(APPEND GREF_SRCS source/graph_ref/floor.c)
endif()

if(CONFIG_GRAPH_REFERENCE_FSMN)
    list(APPEND GREF_SRCS source/graph_ref/fsmn.c)
endif()

if(CONFIG_GRAPH_REFERENCE_FULLYCONNECTED)
    list(APPEND GREF_SRCS source/graph_ref/fullyconnected.c)
endif()

if(CONFIG_GRAPH_REFERENCE_GATHER_ND)
    list(APPEND GREF_SRCS source/graph_ref/gather_nd.c)
endif()

if(CONFIG_GRAPH_REFERENCE_GATHER)
    list(APPEND GREF_SRCS source/graph_ref/gather.c)
endif()

if(CONFIG_GRAPH_REFERENCE_GLOBAL_AVERAGEPOOL)
    list(APPEND GREF_SRCS source/graph_ref/global_averagepool.c)
endif()

if(CONFIG_GRAPH_REFERENCE_GLOBAL_MAXPOOL)
    list(APPEND GREF_SRCS source/graph_ref/global_maxpool.c)
endif()

if(CONFIG_GRAPH_REFERENCE_GREATER_EQUAL)
    list(APPEND GREF_SRCS source/graph_ref/greater_equal.c)
endif()

if(CONFIG_GRAPH_REFERENCE_GREATER)
    list(APPEND GREF_SRCS source/graph_ref/greater.c)
endif()

if(CONFIG_GRAPH_REFERENCE_HARD_SIGMOID)
    list(APPEND GREF_SRCS source/graph_ref/hard_sigmoid.c)
endif()

if(CONFIG_GRAPH_REFERENCE_IM2COL)
    list(APPEND GREF_SRCS source/graph_ref/im2col.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ISNAN)
    list(APPEND GREF_SRCS source/graph_ref/isnan.c)
endif()

if(CONFIG_GRAPH_REFERENCE_L2_NORMALIZATION)
    list(APPEND GREF_SRCS source/graph_ref/l2_normalization.c)
endif()

if(CONFIG_GRAPH_REFERENCE_L2POOL)
    list(APPEND GREF_SRCS source/graph_ref/l2pool.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LAYER_NORMAL)
    list(APPEND GREF_SRCS source/graph_ref/layer_norm.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LEAKY_RELU)
    list(APPEND GREF_SRCS source/graph_ref/leaky_relu.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LESS_EQUAL)
    list(APPEND GREF_SRCS source/graph_ref/less_equal.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LESS)
    list(APPEND GREF_SRCS source/graph_ref/less.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LOG_SOFTMAX)
    list(APPEND GREF_SRCS source/graph_ref/log_softmax.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LOG)
    list(APPEND GREF_SRCS source/graph_ref/log.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LOG1P)
    list(APPEND GREF_SRCS source/graph_ref/log1p.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LOGICAL_AND)
    list(APPEND GREF_SRCS source/graph_ref/logical_and.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LOGICAL_NOT)
    list(APPEND GREF_SRCS source/graph_ref/logical_not.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LOGICAL_OR)
    list(APPEND GREF_SRCS source/graph_ref/logical_or.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LOGICAL_XOR)
    list(APPEND GREF_SRCS source/graph_ref/logical_xor.c)
endif()

if(CONFIG_GRAPH_REFERENCE_LRN)
    list(APPEND GREF_SRCS source/graph_ref/lrn.c)
endif()

if(CONFIG_GRAPH_REFERENCE_MATMUL)
    list(APPEND GREF_SRCS source/graph_ref/matmul.c)
endif()

if(CONFIG_GRAPH_REFERENCE_MAX)
    list(APPEND GREF_SRCS source/graph_ref/max.c)
endif()

if(CONFIG_GRAPH_REFERENCE_MAXIMUM)
    list(APPEND GREF_SRCS source/graph_ref/maximum.c)
endif()

if(CONFIG_GRAPH_REFERENCE_MAXPOOL)
    list(APPEND GREF_SRCS source/graph_ref/maxpool.c)
endif()

if(CONFIG_GRAPH_REFERENCE_MAXPOOL2D_LOCAT)
    list(APPEND GREF_SRCS source/graph_ref/maxpool2d_locat.c)
endif()

if(CONFIG_GRAPH_REFERENCE_MAXPOOL3D)
    list(APPEND GREF_SRCS source/graph_ref/maxpool3d.c)
endif()

if(CONFIG_GRAPH_REFERENCE_MEAN)
    list(APPEND GREF_SRCS source/graph_ref/mean.c)
endif()

if(CONFIG_GRAPH_REFERENCE_MIN)
    list(APPEND GREF_SRCS source/graph_ref/min.c)
endif()

if(CONFIG_GRAPH_REFERENCE_MINIMUM)
    list(APPEND GREF_SRCS source/graph_ref/minimum.c)
endif()

if(CONFIG_GRAPH_REFERENCE_MOD)
    list(APPEND GREF_SRCS source/graph_ref/mod.c)
endif()

if(CONFIG_GRAPH_REFERENCE_MUL)
    list(APPEND GREF_SRCS source/graph_ref/mul.c)
endif()

if(CONFIG_GRAPH_REFERENCE_NDARRAY_SIZE)
    list(APPEND GREF_SRCS source/graph_ref/ndarray_size.c)
endif()

if(CONFIG_GRAPH_REFERENCE_NEGATIVE)
    list(APPEND GREF_SRCS source/graph_ref/negative.c)
endif()

if(CONFIG_GRAPH_REFERENCE_NON_MAX_SUPPRESSION)
    list(APPEND GREF_SRCS source/graph_ref/non_max_suppression.c)
endif()

if(CONFIG_GRAPH_REFERENCE_NOT_EQUAL)
    list(APPEND GREF_SRCS source/graph_ref/not_equal.c)
endif()

if(CONFIG_GRAPH_REFERENCE_NOT)
    list(APPEND GREF_SRCS source/graph_ref/not.c)
endif()

if(CONFIG_GRAPH_REFERENCE_NOT)
    list(APPEND GREF_SRCS source/graph_ref/not.c)
endif()

if(CONFIG_GRAPH_REFERENCE_OR)
    list(APPEND GREF_SRCS source/graph_ref/or.c)
endif()

if(CONFIG_GRAPH_REFERENCE_PAD)
    list(APPEND GREF_SRCS source/graph_ref/pad.c)
endif()

if(CONFIG_GRAPH_REFERENCE_POWER)
    list(APPEND GREF_SRCS source/graph_ref/power.c)
endif()

if(CONFIG_GRAPH_REFERENCE_PRELU)
    list(APPEND GREF_SRCS source/graph_ref/prelu.c)
endif()

if(CONFIG_GRAPH_REFERENCE_PROD)
    list(APPEND GREF_SRCS source/graph_ref/prod.c)
endif()

if(CONFIG_GRAPH_REFERENCE_PROPOSAL)
    list(APPEND GREF_SRCS source/graph_ref/proposal.c)
endif()

if(CONFIG_GRAPH_REFERENCE_PSROIPOOLING)
    list(APPEND GREF_SRCS source/graph_ref/psroipooling.c)
endif()

if(CONFIG_GRAPH_REFERENCE_REDUCE_LOGSUMEXP)
    list(APPEND GREF_SRCS source/graph_ref/reduce_logsumexp.c)
endif()

if(CONFIG_GRAPH_REFERENCE_REDUCE_MAX)
    list(APPEND GREF_SRCS source/graph_ref/reduce_max.c)
endif()

if(CONFIG_GRAPH_REFERENCE_REDUCE_MEAN)
    list(APPEND GREF_SRCS source/graph_ref/reduce_mean.c)
endif()

if(CONFIG_GRAPH_REFERENCE_REDUCE_MIN)
    list(APPEND GREF_SRCS source/graph_ref/reduce_min.c)
endif()

if(CONFIG_GRAPH_REFERENCE_REDUCE_PROD)
    list(APPEND GREF_SRCS source/graph_ref/reduce_prod.c)
endif()

if(CONFIG_GRAPH_REFERENCE_REDUCE_SUM)
    list(APPEND GREF_SRCS source/graph_ref/reduce_sum.c)
endif()

if(CONFIG_GRAPH_REFERENCE_RELU)
    list(APPEND GREF_SRCS source/graph_ref/relu.c)
endif()

if(CONFIG_GRAPH_REFERENCE_RELU1)
    list(APPEND GREF_SRCS source/graph_ref/relu1.c)
endif()

if(CONFIG_GRAPH_REFERENCE_RELU6)
    list(APPEND GREF_SRCS source/graph_ref/relu6.c)
endif()

if(CONFIG_GRAPH_REFERENCE_RELUN)
    list(APPEND GREF_SRCS source/graph_ref/relun.c)
endif()

if(CONFIG_GRAPH_REFERENCE_RESHAPE)
    list(APPEND GREF_SRCS source/graph_ref/reshape.c)
endif()

if(CONFIG_GRAPH_REFERENCE_RESIZE)
    list(APPEND GREF_SRCS source/graph_ref/resize.c)
endif()

if(CONFIG_GRAPH_REFERENCE_REVERSE)
    list(APPEND GREF_SRCS source/graph_ref/reverse.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ROIALIGN)
    list(APPEND GREF_SRCS source/graph_ref/roialign.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ROIPOOL)
    list(APPEND GREF_SRCS source/graph_ref/roipool.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ROUND)
    list(APPEND GREF_SRCS source/graph_ref/round.c)
endif()

if(CONFIG_GRAPH_REFERENCE_RSQRT)
    list(APPEND GREF_SRCS source/graph_ref/rsqrt.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SCATTER)
    list(APPEND GREF_SRCS source/graph_ref/scatter.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SEGMENT_MAX)
    list(APPEND GREF_SRCS source/graph_ref/segment_max.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SEGMENT_MEAN)
    list(APPEND GREF_SRCS source/graph_ref/segment_mean.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SEGMENT_MIN)
    list(APPEND GREF_SRCS source/graph_ref/segment_min.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SEGMENT_PROD)
    list(APPEND GREF_SRCS source/graph_ref/segment_prod.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SEGMENT_SUM)
    list(APPEND GREF_SRCS source/graph_ref/segment_sum.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SELECT)
    list(APPEND GREF_SRCS source/graph_ref/select.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SEQUENCE_MASK)
    list(APPEND GREF_SRCS source/graph_ref/sequence_mask.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SHAPE)
    list(APPEND GREF_SRCS source/graph_ref/shape.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SHUFFLE_CHANNEL)
    list(APPEND GREF_SRCS source/graph_ref/shuffle_channel.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SIGMOID)
    list(APPEND GREF_SRCS source/graph_ref/sigmoid.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SIGN)
    list(APPEND GREF_SRCS source/graph_ref/sign.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SIN)
    list(APPEND GREF_SRCS source/graph_ref/sin.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SINH)
    list(APPEND GREF_SRCS source/graph_ref/sinh.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SLICE)
    list(APPEND GREF_SRCS source/graph_ref/slice.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SOFTMAX)
    list(APPEND GREF_SRCS source/graph_ref/softmax.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SOFTPLUS)
    list(APPEND GREF_SRCS source/graph_ref/softplus.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SOFTRELU)
    list(APPEND GREF_SRCS source/graph_ref/softrelu.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SOFTSIGN)
    list(APPEND GREF_SRCS source/graph_ref/softsign.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SPACE_TO_BATCH)
    list(APPEND GREF_SRCS source/graph_ref/space_to_batch.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SPACE_TO_BATCH_ND)
    list(APPEND GREF_SRCS source/graph_ref/space_to_batch_nd.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SPACE_TO_DEPTH)
    list(APPEND GREF_SRCS source/graph_ref/space_to_depth.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SPLIT)
    list(APPEND GREF_SRCS source/graph_ref/split.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SQRT)
    list(APPEND GREF_SRCS source/graph_ref/sqrt.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SQUARE)
    list(APPEND GREF_SRCS source/graph_ref/square.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SQUEEZE)
    list(APPEND GREF_SRCS source/graph_ref/squeeze.c)
endif()

if(CONFIG_GRAPH_REFERENCE_STACK)
    list(APPEND GREF_SRCS source/graph_ref/stack.c)
endif()

if(CONFIG_GRAPH_REFERENCE_STRIDED_SLICE)
    list(APPEND GREF_SRCS source/graph_ref/strided_slice.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SUB)
    list(APPEND GREF_SRCS source/graph_ref/sub.c)
endif()

if(CONFIG_GRAPH_REFERENCE_SUM)
    list(APPEND GREF_SRCS source/graph_ref/sum.c)
endif()

if(CONFIG_GRAPH_REFERENCE_TAN)
    list(APPEND GREF_SRCS source/graph_ref/tan.c)
endif()

if(CONFIG_GRAPH_REFERENCE_TANH)
    list(APPEND GREF_SRCS source/graph_ref/tanh.c)
endif()

if(CONFIG_GRAPH_REFERENCE_THRESHOLD_RELU)
    list(APPEND GREF_SRCS source/graph_ref/threshold_relu.c)
endif()

if(CONFIG_GRAPH_REFERENCE_TILE)
    list(APPEND GREF_SRCS source/graph_ref/tile.c)
endif()

if(CONFIG_GRAPH_REFERENCE_TOPK)
    list(APPEND GREF_SRCS source/graph_ref/topk.c)
endif()

if(CONFIG_GRAPH_REFERENCE_TRANSPOSE)
    list(APPEND GREF_SRCS source/graph_ref/transpose.c)
endif()

if(CONFIG_GRAPH_REFERENCE_TRUNC)
    list(APPEND GREF_SRCS source/graph_ref/trunc.c)
endif()

if(CONFIG_GRAPH_REFERENCE_UNPOOLING)
    list(APPEND GREF_SRCS source/graph_ref/unpooling.c)
endif()

if(CONFIG_GRAPH_REFERENCE_UNSTACK)
    list(APPEND GREF_SRCS source/graph_ref/unstack.c)
endif()

if(CONFIG_GRAPH_REFERENCE_WHERE)
    list(APPEND GREF_SRCS source/graph_ref/where.c)
endif()

if(CONFIG_GRAPH_REFERENCE_XOR)
    list(APPEND GREF_SRCS source/graph_ref/xor.c)
endif()

if(CONFIG_GRAPH_REFERENCE_YUV_RGB_SCALE)
    list(APPEND GREF_SRCS source/graph_ref/yuv_rgb_scale.c)
endif()

if(CONFIG_GRAPH_REFERENCE_ONE_HOT)
    list(APPEND GREF_SRCS source/graph_ref/one_hot.c)
endif()
