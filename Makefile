CROSS_COMPILE   ?= csky-abiv2-elf-
INSTALL_DIR 	= ../../lib/
NN2_ROOT := $(shell pwd)

ifeq ($(GCOV),y)
        EXTRA_CFLAGS = -fprofile-arcs -ftest-coverage -g -O0
        LIBS   += -fprofile-arcs -ftest-coverage -lgcov
else
        EXTRA_CFLAGS = -O2 -g -Werror -DCSI_DEBUG
endif

export CROSS_COMPILE INSTALL_DIR


all: nn2_ref_x86

nn2_c860:
	DSP_LIB="libcsi_nn2_c860" CFLAGS="-mcpu=c860v -DCSI_BUILD_REF $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="csky-abiv2-linux-" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_c860 -j8
	cd source/; find . -name *.o | xargs rm; cd -

nn2_c906:
	DSP_LIB="libcsi_nn2_c906" CFLAGS="-march=rv64gcvxthead -mabi=lp64dv -DCSI_BUILD_C906 -DCSI_BUILD_REF -DCSI_BUILD_GREF $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="riscv64-unknown-linux-gnu-" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_c906 -j8
	cd source/; find . -name *.o | xargs rm; cd -

nn2_ref_x86:
	DSP_LIB="libcsi_nn2_ref_x86" CFLAGS="$(EXTRA_CFLAGS) -DCSI_BUILD_REF -fPIC -DCSI_AVX_OPT -mavx -mfma -fopenmp" \
	CROSS_COMPILE="" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_ref -j8
	cd source/; find . -name *.o | xargs rm; cd -
	DSP_LIB="libcsi_nn2_ref_x86" CFLAGS="$(EXTRA_CFLAGS) -DCSI_BUILD_REF -fPIC -DCSI_AVX_OPT -mavx -mfma -fopenmp" \
	CROSS_COMPILE="" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_ref nn2_shared -j8
	cd source/; find . -name *.o | xargs rm; cd -

nn2_ref_i805:
	DSP_LIB="libcsi_nn2_ref_i805.a" CFLAGS="-DCSI_BUILD_REF_I805 -DCSI_MATH_DSP -mcpu=i805 $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="csky-abiv2-elf-" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_ref_i805 -j8
	cd source/; find . -name *.o | xargs rm; cd -

nn2_e804:
	DSP_LIB="libcsi_nn2_e804.a" CFLAGS="-DCSI_BUILD_E804 -mcpu=e804d -mno-required-attr-fpu-abi $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="csky-abiv2-elf-" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_e804 -j8
	cd source/; find . -name *.o | xargs rm; cd -

nn2_i805:
	DSP_LIB="libcsi_nn2_i805.a" CFLAGS="-DCSI_BUILD_I805 -DCSI_BUILD_REF -DCSI_BUILD_GREF -mcpu=ck805ef -mhard-float $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="csky-abiv2-elf-" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_i805 -j8
	cd source/; find . -name *.o | xargs rm; cd -


.PHONY: install_nn2
install_nn2: include
	mkdir -p install_nn2/lib
	cp include install_nn2 -r
	cp lib/libcsi_nn2_* install_nn2/lib -rf
	cp version install_nn2/ -rf


clean:
	rm lib/* -rf
	find . -name *.o | xargs rm -rf
