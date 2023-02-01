
if(CONFIG_E907_OPT_SOURCE)
    list(APPEND E907_SRCS source/e907_opt/setup.c)
    list(APPEND E907_SRCS source/e907_opt/utils.c)
endif()

if(CONFIG_E907_OPT_CONVOLUTION)
    list(APPEND E907_SRCS source/e907_opt/convolution.c)
endif()

if(CONFIG_E907_OPT_CONCAT)
    list(APPEND E907_SRCS source/e907_opt/concat.c)
endif()

if(CONFIG_E907_OPT_RELU)
    list(APPEND E907_SRCS source/e907_opt/relu.c)
endif()

if(CONFIG_E907_OPT_FC)
    list(APPEND E907_SRCS source/e907_opt/fullyconnected.c)
    list(APPEND E907_SRCS source/e907_opt/fullyconnected_int8.c)
endif()

if(CONFIG_E907_OPT_MUL)
    list(APPEND E907_SRCS source/e907_opt/mul.c)
endif()

if(CONFIG_E907_OPT_SUM)
    list(APPEND E907_SRCS source/e907_opt/sum.c)
endif()

if(CONFIG_E907_OPT_SOFTMAX)
    list(APPEND E907_SRCS source/e907_opt/softmax.c)
endif()
