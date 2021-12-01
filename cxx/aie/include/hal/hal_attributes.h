/*
 * Copyright (C) 2020-2021 Alibaba Group Holding Limited
 */
#ifndef HAL_ATTRIBUTES_H
#define HAL_ATTRIBUTES_H

#ifdef __GNUC__
#    define GCC_VERSION_AT_LEAST(x,y) (__GNUC__ > (x) || __GNUC__ == (x) && __GNUC_MINOR__ >= (y))
#    define GCC_VERSION_AT_MOST(x,y)  (__GNUC__ < (x) || __GNUC__ == (x) && __GNUC_MINOR__ <= (y))
#else
#    define GCC_VERSION_AT_LEAST(x,y) 0
#    define GCC_VERSION_AT_MOST(x,y)  0
#endif

//use '-Wno-deprecated-declarations' to ignore warning
#if GCC_VERSION_AT_LEAST(3,1)
#    define attribute_deprecated __attribute__((deprecated))
#elif defined(_MSC_VER)
#    define attribute_deprecated __declspec(deprecated)
#else
#    define attribute_deprecated
#endif

#endif  // HAL_ATTRIBUTES_H
