# Release or Debug or MinSizeRel
BUILD_TYPE=Release
USE_CORE=8
INSTALL_DIR=../install_nn2/
all: nn2_ref_x86

nn2_e907_elf:
	mkdir -p e907_build; cd e907_build; cmake ../ -DCONFIG_BUILD_RISCV_ELF_E907=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/e907/; make -j${USE_CORE}; make install; cd -

nn2_rvv:
	mkdir -p rvv_build; cd rvv_build; cmake ../ -DCONFIG_BUILD_RISCV_RVV=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/rvv/; make -j${USE_CORE}; make install; cd -

nn2_rvv_nodot:
	mkdir -p rvv_nodot_build; cd rvv_nodot_build; cmake ../ -DCONFIG_BUILD_RISCV_RVV_NODOT=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/rvv_nodot/; make -j${USE_CORE}; make install; cd -

nn2_c906:
	mkdir -p c906_static_build; cd c906_static_build; cmake ../ -DCONFIG_BUILD_RISCV_C906=ON -DCONFIG_SHL_BUILD_STATIC=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/c906/; make -j${USE_CORE}; make install; cd -

nn2_c906_so:
	mkdir -p c906_so_build; cd c906_so_build; cmake ../ -DCONFIG_BUILD_RISCV_C906=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/c906/; make -j${USE_CORE}; make install; cd -

nn2_c906_elf:
	mkdir -p c906_elf_build; cd c906_elf_build; cmake ../ -DCONFIG_BUILD_RISCV_ELF_C906=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/c906/; make -j${USE_CORE}; make install; cd -

nn2_rvm:
	mkdir -p rvm_build; cd rvm_build; cmake ../ -DCONFIG_BUILD_RISCV_RVM=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/rvm/; make -j${USE_CORE}; make install; cd -

nn2_c908:
	mkdir -p c908_build; cd c908_build; cmake ../ -DCONFIG_BUILD_RISCV_C908=ON -DCONFIG_SHL_BUILD_STATIC=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/c908/; make -j${USE_CORE}; make install; cd -

nn2_c920:
	mkdir -p c920_build; cd c920_build; cmake ../ -DCONFIG_BUILD_RISCV_C920=ON -DCONFIG_SHL_BUILD_STATIC=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/c920/; make -j${USE_CORE}; make install; cd -

nn2_c920v2:
	mkdir -p c920v2_build; cd c920v2_build; cmake ../ -DCONFIG_BUILD_RISCV_C920V2=ON -DCONFIG_SHL_BUILD_STATIC=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/c920v2/; make -j${USE_CORE}; make install; cd -

nn2_c920_so:
	mkdir -p c920_build_so; cd c920_build_so; cmake ../ -DCONFIG_BUILD_RISCV_C920=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/c920/; make -j${USE_CORE}; make install; cd -

nn2_ref_x86:
	mkdir -p x86_ref_build; cd x86_ref_build; cmake ../ -DCONFIG_BUILD_X86_REF=ON -DCONFIG_SHL_BUILD_STATIC=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/x86/; make -j${USE_CORE}; make install; cd -

nn2_ref_x86_so:
	mkdir -p x86_ref_build_so; cd x86_ref_build_so; cmake ../ -DCONFIG_BUILD_X86_REF=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/x86/; make -j${USE_CORE}; make install; cd -

menuconfig:
	env  KCONFIG_BASE=    python3  script/kconfig/menuconfig.py Kconfig

.PHONY: install_nn2
install_nn2: include
	cp version install_nn2/ -rf

clint:
	./script/git-clang-format.sh origin/master

clean:
	rm lib/* *_build *_build_so install_nn2 -rf
