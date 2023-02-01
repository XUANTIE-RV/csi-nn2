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

if(CONFIG_BUILD_X86_REF)
    # build x86_ref so
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/x86_ref.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND X86_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS})
    add_library(x86_static STATIC ${X86_LST})
    SET_TARGET_PROPERTIES(x86_static PROPERTIES OUTPUT_NAME "shl_ref_x86")
    set(X86_BUILD_FLAGS -DSHL_AVX_OPT -DSHL_BUILD_REF -DSHL_BUILD_GREF -fPIC -mavx -mfma -fopenmp)
    target_compile_options(x86_static PRIVATE ${X86_BUILD_FLAGS})

    install(TARGETS x86_static DESTINATION lib)
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
    set(RVV_BUILD_FLAGS -march=rv64gcv_zfh_xtheadc_xtheadvdot -mabi=lp64d -DSHL_BUILD_RVV -DSHL_BUILD_REF -DSHL_BUILD_GREF)
    target_compile_options(rvv_static PRIVATE ${RVV_BUILD_FLAGS})

    install(TARGETS rvv_static DESTINATION lib)
endif()

if(CONFIG_BUILD_RISCV_C906_STATIC)
    # build c906 a
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/c906_static.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND C906_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS} ${THEAD_RVV_SRCS} ${C906_SRCS})
    add_library(c906_static STATIC ${C906_LST})
    SET_TARGET_PROPERTIES(c906_static PROPERTIES OUTPUT_NAME "shl_c906")
    set(C906_BUILD_FLAGS -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -DSHL_BUILD_C906 -DSHL_BUILD_REF -DSHL_BUILD_GREF -DSHL_BUILD_RVV)
    target_compile_options(c906_static PRIVATE ${C906_BUILD_FLAGS})

    install(TARGETS c906_static DESTINATION lib)
endif()

if(CONFIG_BUILD_RISCV_C906_SHARE)
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/c906_share.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND C906_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS} ${THEAD_RVV_SRCS} ${C906_SRCS})
    add_library(c906_share SHARED ${C906_LST})
    SET_TARGET_PROPERTIES(c906_share PROPERTIES OUTPUT_NAME "shl_c906")
    set(C906_BUILD_FLAGS -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -DSHL_BUILD_C906 -DSHL_BUILD_REF -DSHL_BUILD_GREF -DSHL_BUILD_RVV)
    target_compile_options(c906_share PRIVATE ${C906_BUILD_FLAGS})

    install(TARGETS c906_share DESTINATION lib)
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
    set(RVM_BUILD_FLAGS -march=rv64gcv_zfh_xtheadc_xtheadvdot_xtheadmatrix -mabi=lp64d -DSHL_BUILD_RVM -DSHL_BUILD_REF -DSHL_BUILD_GREF -DSHL_BUILD_RVV)
    target_compile_options(rvm_static PRIVATE ${RVM_BUILD_FLAGS})

    install(TARGETS rvm_static DESTINATION lib)
endif()

if(CONFIG_BUILD_RISCV_C908)
    # build c908 a
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/c908.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND C908_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS} ${THEAD_RVV_SRCS} ${C908_SRCS})
    add_library(c908_static STATIC ${C908_LST})
    SET_TARGET_PROPERTIES(c908_static PROPERTIES OUTPUT_NAME "shl_c908")
    set(C908_BUILD_FLAGS -march=rv64gcv_zfh_xtheadc_xtheadvdot -mabi=lp64d -DSHL_BUILD_C908 -DSHL_BUILD_REF -DSHL_BUILD_GREF -DSHL_BUILD_RVV)
    target_compile_options(c908_static PRIVATE ${C908_BUILD_FLAGS})

    install(TARGETS c908_static DESTINATION lib)
endif()

if(CONFIG_BUILD_RISCV_C910)
    # build c908 a
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/c910.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND C910_LST ${NN2_SRCS} ${REF_SRCS} ${GREF_SRCS} ${THEAD_RVV_SRCS} ${C906_SRCS} ${C910_SRCS})
    add_library(c910_static STATIC ${C910_LST})
    SET_TARGET_PROPERTIES(c910_static PROPERTIES OUTPUT_NAME "shl_c910")
    set(C910_BUILD_FLAGS -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -DSHL_BUILD_C910 -DDSHL_BUILD_C906 -DSHL_BUILD_REF -DSHL_BUILD_GREF -DSHL_BUILD_RVV)
    target_compile_options(c910_static PRIVATE ${C910_BUILD_FLAGS})

    install(TARGETS c910_static DESTINATION lib)
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

if(CONFIG_BUILD_CSKY_C860)
    # build c860 a
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/c860.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND C860_LST ${NN2_SRCS} ${REF_SRCS} ${C860_SRCS})
    add_library(c860_static STATIC ${C860_LST})
    SET_TARGET_PROPERTIES(c860_static PROPERTIES OUTPUT_NAME "shl_c860")
    set(C860_BUILD_FLAGS -mcpu=c860v -DSHL_BUILD_REF)
    target_compile_options(c860_static PRIVATE ${C860_BUILD_FLAGS})

    install(TARGETS c860_static DESTINATION lib)
endif()

if(CONFIG_BUILD_CSKY_ELF_I805_REF)
    # build i805 ref a
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/i805_ref.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND I805_REF_LST ${NN2_SRCS} ${REF_SRCS} ${I805_REF_SRCS})
    add_library(i805_ref_static STATIC ${I805_REF_LST})
    SET_TARGET_PROPERTIES(i805_ref_static PROPERTIES OUTPUT_NAME "shl_ref_i805")
    set(I805_REF_BUILD_FLAGS -DSHL_BUILD_REF_I805 -DSHL_BUILD_RTOS -mcpu=i805)
    target_compile_options(i805_ref_static PRIVATE ${I805_REF_BUILD_FLAGS})
    target_include_directories(i805_ref_static PRIVATE source/i805_ref)

    install(TARGETS i805_ref_static DESTINATION lib)
endif()

if(CONFIG_BUILD_CSKY_ELF_I805)
    # build i805 a
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/i805.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND I805_LST ${NN2_SRCS} ${REF_SRCS} ${I805_SRCS})
    add_library(i805_static STATIC ${I805_LST})
    SET_TARGET_PROPERTIES(i805_static PROPERTIES OUTPUT_NAME "shl_i805")
    set(I805_BUILD_FLAGS -DSHL_BUILD_I805 -DSHL_BUILD_RTOS -mcpu=ck805ef -mhard-float)
    target_compile_options(i805_static PRIVATE ${I805_BUILD_FLAGS})
    target_include_directories(i805_static PRIVATE source/i805_opt)

    install(TARGETS i805_static DESTINATION lib)
endif()

if(CONFIG_BUILD_CSKY_ELF_E804)
    # build e804 a
    if (NOT CONFIG_CUSTOM_SOURCE_SELECT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/e804.cmake)
    endif()
    include(cmake/rules.cmake)
    LIST(APPEND E804_LST ${NN2_SRCS} ${REF_SRCS} ${E804_SRCS})
    add_library(e804_static STATIC ${E804_LST})
    SET_TARGET_PROPERTIES(e804_static PROPERTIES OUTPUT_NAME "shl_e804")
    set(E804_BUILD_FLAGS -DSHL_BUILD_E804 -mcpu=e804d -DSHL_BUILD_RTOS -mno-required-attr-fpu-abi)
    target_compile_options(e804_static PRIVATE ${E804_BUILD_FLAGS})
    target_include_directories(e804_static PRIVATE source/e804_opt)

    install(TARGETS e804_static DESTINATION lib)
endif()

# coverage options
OPTION(ENABLE_GCOV "Enable gcov" OFF)
if(ENABLE_GCOV)
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
  SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage -lgcov")
endif()


install(DIRECTORY "include/." DESTINATION "include" FILES_MATCHING PATTERN "*.h")
