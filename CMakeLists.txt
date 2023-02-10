cmake_minimum_required(VERSION 3.3)
project(SHL)

enable_language(ASM)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

if (CONFIG_CUSTOM_SOURCE_SELECT)
    add_definitions(-DSHL_MCONF_CONFIG)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
endif()

file (STRINGS "${CMAKE_CURRENT_BINARY_DIR}/../version" SHL_VERSION)
set(SHL_MAJOR_VERSION 2)
SET(so_symlink_target ${CMAKE_INSTALL_PREFIX}/lib/libshl.so)
SET(a_symlink_target ${CMAKE_INSTALL_PREFIX}/lib/libshl.a)

if(CONFIG_BUILD_X86_REF)
    # build x86_ref lib
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/x86_ref.cmake)
    endif()
    include(cmake/rules.cmake)

    set(SHL_LIB_TARGET "ref_x86_target")
    set(SHL_LIB_NAME shl_ref_x86)
    LIST(APPEND SHL_BUILD_SRC_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS})
    set(SHL_BUILD_C_FLAGS -DSHL_AVX_OPT -DSHL_BUILD_REF -DSHL_BUILD_GREF -fPIC -mavx -mfma -fopenmp)
    include(cmake/target_build.cmake)
endif()

if(CONFIG_BUILD_RISCV_RVV)
    # build rvv a
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rvv.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND RVV_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS} ${THEAD_RVV_SRCS})
    add_library(rvv_static STATIC ${RVV_LST})
    SET_TARGET_PROPERTIES(rvv_static PROPERTIES OUTPUT_NAME "shl_rvv")
    set(RVV_BUILD_FLAGS -ffp-contract=off -march=rv64gcv_zfh_xtheadc_xtheadvdot -mabi=lp64d -DSHL_BUILD_RVV -DSHL_BUILD_REF -DSHL_BUILD_GREF)
    target_compile_options(rvv_static PRIVATE ${RVV_BUILD_FLAGS})

    install(TARGETS rvv_static DESTINATION lib)
endif()

if(CONFIG_BUILD_RISCV_C906)
    # build c906 lib
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/c906_static.cmake)
    endif()
    include(cmake/rules.cmake)

    set(SHL_LIB_TARGET "c906_lib")
    set(SHL_LIB_NAME shl_c906)
    LIST(APPEND SHL_BUILD_SRC_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS} ${THEAD_RVV_SRCS} ${C906_SRCS})
    set(SHL_BUILD_C_FLAGS -ffp-contract=off -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -DSHL_BUILD_C906 -DSHL_BUILD_REF -DSHL_BUILD_GREF -DSHL_BUILD_RVV)
    include(cmake/target_build.cmake)
    target_include_directories(${SHL_LIB_TARGET} PRIVATE module/dlpack/include/)
endif()

if(CONFIG_BUILD_RISCV_RVM)
    # build rvm a
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rvm.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND RVM_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS} ${THEAD_RVV_SRCS} ${THEAD_MATRIX_SRCS} ${RVM_SRCS})
    add_library(rvm_static STATIC ${RVM_LST})
    SET_TARGET_PROPERTIES(rvm_static PROPERTIES OUTPUT_NAME "shl_rvm")
    set(RVM_BUILD_FLAGS -ffp-contract=off -march=rv64gcv_zfh_xtheadc_xtheadvdot_xtheadmatrix -mabi=lp64d -DSHL_BUILD_RVM -DSHL_BUILD_REF -DSHL_BUILD_GREF -DSHL_BUILD_RVV)
    target_compile_options(rvm_static PRIVATE ${RVM_BUILD_FLAGS})

    install(TARGETS rvm_static DESTINATION lib)
endif()

if(CONFIG_BUILD_RISCV_C908)
    # build c908 lib
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/c908.cmake)
    endif()
    include(cmake/rules.cmake)

    set(SHL_LIB_TARGET "c908_lib")
    set(SHL_LIB_NAME shl_c908)
    LIST(APPEND SHL_BUILD_SRC_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS} ${THEAD_RVV_SRCS} ${C908_SRCS})
    set(SHL_BUILD_C_FLAGS -ffp-contract=off -march=rv64gcv_zfh_xtheadc_xtheadvdot -mabi=lp64d -DSHL_BUILD_C908 -DSHL_BUILD_REF -DSHL_BUILD_GREF -DSHL_BUILD_RVV)
    include(cmake/target_build.cmake)
endif()

if(CONFIG_BUILD_RISCV_C920)
    # build c920 lib
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/c920.cmake)
    endif()
    include(cmake/rules.cmake)

    set(SHL_LIB_TARGET "c920_lib")
    set(SHL_LIB_NAME shl_c920)
    LIST(APPEND SHL_BUILD_SRC_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS} ${THEAD_RVV_SRCS} ${C920_SRCS})
    set(SHL_BUILD_C_FLAGS -ffp-contract=off -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -DSHL_BUILD_C920 -DSHL_BUILD_REF -DSHL_BUILD_GREF -DSHL_BUILD_RVV)
    include(cmake/target_build.cmake)
    target_include_directories(${SHL_LIB_TARGET} PRIVATE module/dlpack/include/)
endif()

if(CONFIG_BUILD_RISCV_ELF_C906)
    # build c906 elf a
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/c906_elf.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND C906_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS} ${THEAD_RVV_SRCS} ${C906_SRCS})
    add_library(c906_elf_static STATIC ${C906_LST})
    SET_TARGET_PROPERTIES(c906_elf_static PROPERTIES OUTPUT_NAME "shl_c906_rtos")
    set(C906_BUILD_FLAGS -ffp-contract=off -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mcmodel=medany -DSHL_BUILD_C906 -DSHL_BUILD_REF -DSHL_BUILD_GREF -DSHL_BUILD_RTOS -DSHL_BUILD_RVV)
    target_compile_options(c906_elf_static PRIVATE ${C906_BUILD_FLAGS})

    install(TARGETS c906_elf_static DESTINATION lib)
endif()

if(CONFIG_BUILD_RISCV_ELF_E907)
    # build e907/e906 elf a
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/e907.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND E907_LST ${NN2_SRCS} ${REF_SRCS} ${E907_SRCS} ${GREF_SRCS})
    add_library(e907_elf_static STATIC ${E907_LST})
    SET_TARGET_PROPERTIES(e907_elf_static PROPERTIES OUTPUT_NAME "shl_e907")
    set(E907_BUILD_FLAGS -march=rv32imafcpzpsfoperand_xtheade -mabi=ilp32f -mcmodel=medlow -g2 -Os -DSHL_BUILD_REF -DSHL_BUILD_GREF -DSHL_BUILD_E907 -DSHL_BUILD_RTOS)
    target_compile_options(e907_elf_static PRIVATE ${E907_BUILD_FLAGS})
    target_include_directories(e907_elf_static PRIVATE module/)

    install(TARGETS e907_elf_static DESTINATION lib)
endif()

# coverage options
OPTION(ENABLE_GCOV "Enable gcov" OFF)
if(ENABLE_GCOV)
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
  SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage -lgcov")
endif()


install(DIRECTORY "include/." DESTINATION "include" FILES_MATCHING PATTERN "*.h")
