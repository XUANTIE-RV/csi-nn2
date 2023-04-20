# Release or Debug or MinSizeRel
BUILD_TYPE=Release
USE_CORE=8
INSTALL_DIR=../install_nn2/
all: nn2_ref_x86

nn2_c860:
	mkdir -p c860_build; cd c860_build; cmake ../ -DCONFIG_BUILD_CSKY_C860=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}; make -j${USE_CORE}; make install; cd -

nn2_e907_elf:
	mkdir -p e907_build; cd e907_build; cmake ../ -DCONFIG_BUILD_RISCV_ELF_E907=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}; make -j${USE_CORE}; make install; cd -

nn2_rvv:
	mkdir -p rvv_build; cd rvv_build; cmake ../ -DCONFIG_BUILD_RISCV_RVV=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}; make -j${USE_CORE}; make install; cd -

nn2_c906:
	mkdir -p c906_static_build; cd c906_static_build; cmake ../ -DCONFIG_BUILD_RISCV_C906_STATIC=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}; make -j${USE_CORE}; make install; cd -

nn2_c906_so:
	mkdir -p c906_so_build; cd c906_so_build; cmake ../ -DCONFIG_BUILD_RISCV_C906_SHARE=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}; make -j${USE_CORE}; make install; cd -

nn2_c906_elf:
	mkdir -p c906_elf_build; cd c906_elf_build; cmake ../ -DCONFIG_BUILD_RISCV_ELF_C906=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}; make -j${USE_CORE}; make install; cd -

nn2_rvm:
	mkdir -p rvm_build; cd rvm_build; cmake ../ -DCONFIG_BUILD_RISCV_RVM=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}; make -j${USE_CORE}; make install; cd -

nn2_c908:
	mkdir -p c908_build; cd c908_build; cmake ../ -DCONFIG_BUILD_RISCV_C908=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}; make -j${USE_CORE}; make install; cd -

nn2_c920:
	mkdir -p c920_build; cd c920_build; cmake ../ -DCONFIG_BUILD_RISCV_C920=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}; make -j${USE_CORE}; make install; cd -

nn2_ref_x86:
	mkdir -p x86_ref_build; cd x86_ref_build; cmake ../ -DCONFIG_BUILD_X86_REF=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}; make -j${USE_CORE}; make install; cd -

menuconfig:
	env  KCONFIG_BASE=    python3  script/kconfig/menuconfig.py Kconfig

.PHONY: install_nn2
install_nn2: include
	cp version install_nn2/ -rf

clint:
	./script/git-clang-format.sh origin/master

clean:
	rm lib/* *_build install_nn2 -rf
