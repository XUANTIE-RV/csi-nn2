if (NOT CONFIG_USE_COMPILER_PATH)

# riscv linux compiler
if (CONFIG_BUILD_RISCV_RVV OR CONFIG_BUILD_RISCV_RVV_NODOT OR
    CONFIG_BUILD_RISCV_C906 OR CONFIG_BUILD_RISCV_RVM OR
    CONFIG_BUILD_RISCV_C908 OR CONFIG_BUILD_RISCV_C920 OR
    CONFIG_BUILD_RISCV_C920V2 OR CONFIG_BUILD_RISCV_PNNA OR
    CONFIG_BUILD_TH1520)
    set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
    set(CMAKE_ASM_COMPILER riscv64-unknown-linux-gnu-gcc)
endif()

# riscv elf compiler
if (CONFIG_BUILD_RISCV_ELF_C906 OR CONFIG_BUILD_RISCV_ELF_E907)
    set(CMAKE_ASM_COMPILER riscv64-unknown-elf-gcc)
    set(CMAKE_C_COMPILER riscv64-unknown-elf-gcc)
    set(CMAKE_CXX_COMPILER riscv64-unknown-elf-gcc)
endif()

endif()

# SHL debug module
if(CONFIG_USE_SHL_DEBUG)
    add_definitions(-D SHL_DEBUG)
endif()

# SHL export model
if(CONFIG_USE_EXPORT_MODEL)
    add_definitions(-D SHL_EXPORT_MODEL)
endif()

if (CONFIG_BUILD_ANDROID_TH1520)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBUILD_ANDROID -Wno-deprecated-non-prototype")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_ANDROID")
else()
# reduce elf size
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections -Wl,--gc-sections")
endif()

# set warning as error
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")

file(GLOB_RECURSE NN2_SRCS source/nn2/*.c source/utils/*.c source/utils/*.cpp)
file(GLOB_RECURSE REF_SRCS source/reference/*.c)
file(GLOB_RECURSE GREF_SRCS source/graph_ref/*.c)
file(GLOB_RECURSE THEAD_RVV_SRCS source/thead_rvv/*.c)
file(GLOB_RECURSE PNNA_SRCS source/pnna/*.c source/pnna/*.cpp)
file(GLOB_RECURSE THEAD_MATRIX_SRCS source/thead_matrix/*.c source/thead_matrix/*.S)
file(GLOB_RECURSE C906_SRCS source/c906_opt/*.c source/c906_opt/*.S)
file(GLOB_RECURSE C908_SRCS source/c908_opt/*.c source/c908_opt/*.S)
file(GLOB_RECURSE C920_SRCS source/c920_opt/*.c source/c920_opt/*.S)
file(GLOB_RECURSE C920V2_SRCS source/c920v2_opt/*.c source/c920v2_opt/*.S)
file(GLOB_RECURSE LLM_SRCS source/llm/*.c source/llm/*.cpp)

include(source/reference/CMakeLists.txt)
include(source/graph_ref/CMakeLists.txt)
include(source/thead_rvv/CMakeLists.txt)
include(source/c906_opt/CMakeLists.txt)
include(source/e907_opt/CMakeLists.txt)

include_directories(include include/csinn include/graph include/backend module)

if(CONFIG_SHL_LAYER_BENCHMARK)
    add_definitions(-DSHL_LAYER_BENCHMARK)
    message(STATUS "Print the execution time of each layer - ON")
endif()

if(CONFIG_GRAPH_REFERENCE_TVMGEN)
    add_definitions(-DGRAPH_REFERENCE_TVMGEN)
    LIST(APPEND GREF_SRCS source/tvm_gen/utils.c source/tvm_gen/setup.c)
endif()
