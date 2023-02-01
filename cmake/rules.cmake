if (NOT CONFIG_USE_COMPILER_PATH)

# riscv linux compiler
if (CONFIG_BUILD_RISCV_RVV OR CONFIG_BUILD_RISCV_C906_STATIC OR
    CONFIG_BUILD_RISCV_C906_SHARE OR CONFIG_BUILD_RISCV_RVM OR
    CONFIG_BUILD_RISCV_C908 OR CONFIG_BUILD_RISCV_C910)
    set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
    set(CMAKE_ASM_COMPILER riscv64-unknown-linux-gnu-gcc)
endif()

# riscv elf compiler
if (CONFIG_BUILD_RISCV_ELF_C906 OR CONFIG_BUILD_RISCV_ELF_E907 OR
    CONFIG_BUILD_RISCV_ELF_ASP)
    set(CMAKE_ASM_COMPILER riscv64-unknown-elf-gcc)
    set(CMAKE_C_COMPILER riscv64-unknown-elf-gcc)
endif()

# csky linux compiler
if (CONFIG_BUILD_CSKY_OPENVX OR CONFIG_BUILD_CSKY_C860)
    set(CMAKE_C_COMPILER csky-abiv2-linux-gcc)
    set(CMAKE_ASM_COMPILER csky-abiv2-linux-gcc)
endif()

# csky elf compiler
if (CONFIG_BUILD_CSKY_ELF_I805_REF OR CONFIG_BUILD_CSKY_ELF_I805 OR
    CONFIG_BUILD_CSKY_ELF_E804)
    set(CMAKE_C_COMPILER csky-abiv2-elf-gcc)
    set(CMAKE_ASM_COMPILER csky-abiv2-elf-gcc)
endif()

endif()

# SHL debug module
if(CONFIG_USE_SHL_DEBUG)
    add_definitions(-D SHL_DEBUG)
endif()

# reduce elf size
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections -Wl,--gc-sections")

# set warning as error
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")

file(GLOB_RECURSE NN2_SRCS source/nn2/*.c source/utils/*.c)
include(source/reference/CMakeLists.txt)
include(source/graph_ref/CMakeLists.txt)
file(GLOB_RECURSE C906_SRCS source/c906_opt/*.c source/c906_opt/*.S)
file(GLOB_RECURSE C908_SRCS source/c908_opt/*.c source/c908_opt/gemm_kernel/*.S)
file(GLOB_RECURSE THEAD_RVV_SRCS source/thead_rvv/*.c)
file(GLOB_RECURSE THEAD_MATRIX_SRCS source/thead_matrix/*.c source/thead_matrix/*.S)
file(GLOB_RECURSE C860_SRCS source/c860_opt/*.S)
file(GLOB_RECURSE I805_REF_SRCS source/i805_ref/*.c)
file(GLOB_RECURSE I805_SRCS source/i805_opt/*.c source/i805_opt/*.S)
file(GLOB_RECURSE E804_SRCS source/e804_opt/*.c source/e804_opt/*.S)
file(GLOB_RECURSE ASP_SRCS source/asp/*.c)
file(GLOB_RECURSE C910_SRCS source/c910_opt/*.c source/c910_opt/gemm_kernel/*.S)
include(source/e907_opt/CMakeLists.txt)

include_directories(include)

option(CONFIG_SHL_LAYER_BENCHMARK "Layer information and performance" OFF)
if(CONFIG_SHL_LAYER_BENCHMARK)
    add_definitions(-DSHL_LAYER_BENCHMARK)
    message(STATUS "Print the execution time of each layer - ON")
endif()

if(CONFIG_GRAPH_REFERENCE_SUBGRAPH)
    add_definitions(-DGRAPH_REFERENCE_SUBGRAPH)
endif()
