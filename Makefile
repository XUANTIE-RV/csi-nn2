all: nn2_ref_x86

nn2_c860:
	mkdir -p csky_build; cd csky_build; cmake ../ -DCONFIG_BUILD_CSKY_C860=ON -DCMAKE_BUILD_TYPE=Release; make c860_static -j8; cd -

nn2_rvv:
	mkdir -p riscv_build; cd riscv_build; cmake ../ -DCONFIG_BUILD_RISCV_RVV=ON -DCMAKE_BUILD_TYPE=Release; make rvv_static -j8; cd -

nn2_c906:
	mkdir -p riscv_build; cd riscv_build; cmake ../ -DCONFIG_BUILD_RISCV_C906_STATIC=ON -DCMAKE_BUILD_TYPE=Release; make c906_static -j8; cd -

nn2_c906_so:
	mkdir -p riscv_build; cd riscv_build; cmake ../ -DCONFIG_BUILD_RISCV_C906_SHARE=Release; make c906_share -j8; cd -

nn2_c906_elf:
	mkdir -p riscv_elf_build; cd riscv_elf_build; cmake ../ -DCONFIG_BUILD_RISCV_ELF_C906=ON -DCMAKE_BUILD_TYPE=Release; make c906_elf_static -j8; cd -

nn2_c908:
	mkdir -p riscv_build; cd riscv_build; cmake ../ -DCONFIG_BUILD_RISCV_C908=ON -DCMAKE_BUILD_TYPE=Release; make c908_static -j8; cd -

nn2_rvm:
	mkdir -p riscv_build; cd riscv_build; cmake ../ -DCONFIG_BUILD_RISCV_RVM=ON -DCMAKE_BUILD_TYPE=Release; make rvm_static -j8; cd -

nn2_ref_x86:
	mkdir -p x86_build; cd x86_build; cmake ../ -DCONFIG_BUILD_X86_REF=ON -DCMAKE_BUILD_TYPE=Release; make x86_static -j8; cd -

.PHONY: install_nn2
install_nn2: include
	mkdir -p install_nn2/lib
	cp include install_nn2 -r
	-cp riscv_build/libshl_* install_nn2/lib -rf
	-cp csky_build/libshl_* install_nn2/lib -rf
	-cp x86_build/libshl_* install_nn2/lib -rf
	cp version install_nn2/ -rf

clint:
	./script/git-clang-format.sh origin/master

clean:
	rm lib/*  -rf
