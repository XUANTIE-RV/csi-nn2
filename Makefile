all: nn2_ref_x86

nn2_c860:
	mkdir -p csky_build; cd csky_build; cmake ../ -DBUILD_CSKY=ON -DCMAKE_BUILD_TYPE=Release; make c860_static -j8; cd -

nn2_rvv:
	mkdir -p riscv_build; cd riscv_build; cmake ../ -DBUILD_RISCV=ON -DCMAKE_BUILD_TYPE=Release; make rvv_static -j8; cd -

nn2_c906:
	mkdir -p riscv_build; cd riscv_build; cmake ../ -DBUILD_RISCV=ON -DCMAKE_BUILD_TYPE=Release; make c906_static -j8; cd -

nn2_c906_so:
	mkdir -p riscv_build; cd riscv_build; cmake ../ -DBUILD_RISCV=ON -DCMAKE_BUILD_TYPE=Release; make c906_share -j8; cd -

nn2_c906_elf:
	mkdir -p riscv_elf_build; cd riscv_elf_build; cmake ../ -DBUILD_RISCV_ELF=ON -DCMAKE_BUILD_TYPE=Release; make c906_elf_static -j8; cd -

nn2_asp_elf:
	mkdir -p riscv_elf_build; cd riscv_elf_build; cmake ../ -DBUILD_RISCV_ELF=ON -DCMAKE_BUILD_TYPE=Release; make asp_elf_static -j8; cd -

nn2_c908:
	mkdir -p riscv_build; cd riscv_build; cmake ../ -DBUILD_RISCV=ON -DCMAKE_BUILD_TYPE=Release; make c908_static -j8; cd -

nn2_ref_x86:
	mkdir -p x86_build; cd x86_build; cmake ../ -DBUILD_X86=ON -DCMAKE_BUILD_TYPE=Release; make x86_static -j8; cd -

nn2_openvx:
	mkdir -p csky_build; cd csky_build; cmake ../ -DBUILD_CSKY=ON -DCMAKE_BUILD_TYPE=Release; make openvx_share -j8; cd -

nn2_pnna:
	mkdir -p riscv_build; cd riscv_build; cmake ../ -DBUILD_RISCV=ON -DCMAKE_BUILD_TYPE=Release; make pnna_share -j8; cd -

nn2_pnna_x86:
	mkdir -p x86_build; cd x86_build; cmake ../ -DBUILD_X86=ON -DCMAKE_BUILD_TYPE=Release; make pnna_share -j8; cd -

nn2_hlight_x86:
	mkdir -p x86_build; cd x86_build; cmake ../ -DBUILD_X86=ON -DCMAKE_BUILD_TYPE=Release; make hlight_share -j8; cd -

nn2_hlight:
	mkdir -p riscv_build; cd riscv_build; cmake ../ -DBUILD_RISCV=ON -DCMAKE_BUILD_TYPE=Release; make hlight_share -j8; cd -

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
