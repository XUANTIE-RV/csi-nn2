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


all: nn2_ref_x86 nn2_ref

nn2_ref:
	DSP_LIB="libcsi_nn2_ref" CFLAGS="-mcpu=c860v -DCSI_BUILD_REF $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="csky-abiv2-linux-" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_ref -j8
	cd source/; find . -name *.o | xargs rm; cd -

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

nn2_openvx:
	DSP_LIB="libcsi_nn2_openvx.a" CFLAGS="-mcpu=c860v -DCSI_BUILD_OPENVX -mhard-float $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="csky-abiv2-linux-" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_openvx -j8
	cd source/; find . -name *.o | xargs rm; cd -
	DSP_LIB="libcsi_nn2_openvx.so" CFLAGS="-mcpu=c860v -fPIC -DCSI_BUILD_OPENVX -mhard-float $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="csky-abiv2-linux-" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_openvx nn2_shared -j8
	cd source/; find . -name *.o | xargs rm; cd -

nn2_pnna:
	DSP_LIB="libcsi_nn2_pnna.a" CFLAGS="-DCSI_BUILD_PNNA $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="riscv64-unknown-linux-gnu-" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_pnna -j8
	cd source/; find . -name *.o | xargs rm; cd -
	DSP_LIB="libcsi_nn2_pnna.so" CFLAGS="-fPIC -DCSI_BUILD_PNNA $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="riscv64-unknown-linux-gnu-" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_pnna nn2_shared -j8
	cd source/; find . -name *.o | xargs rm; cd -

nn2_pnna_x86:
	DSP_LIB="libcsi_nn2_pnna_x86.a" CFLAGS="-DCSI_BUILD_PNNA $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_pnna -j8
	cd source/; find . -name *.o | xargs rm; cd -
	DSP_LIB="libcsi_nn2_pnna_x86.so" CFLAGS="-fPIC -DCSI_BUILD_PNNA $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_pnna nn2_shared_x86 -j8
	cd source/; find . -name *.o | xargs rm; cd -

nn2_gref:
	DSP_LIB="libcsi_nn2_gref.a" CFLAGS="-DCSI_BUILD_REF -DCSI_BUILD_GREF $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_gref -j8
	cd source/; find . -name *.o | xargs rm; cd -
	DSP_LIB="libcsi_nn2_gref.so" CFLAGS="-fPIC -DCSI_BUILD_REF -DCSI_BUILD_GREF $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_gref nn2_shared -j8
	cd source/; find . -name *.o | xargs rm; cd -

nn2_dp1k:
	DSP_LIB="libcsi_nn2_dp1000.so" CFLAGS="-fPIC -DCSI_BUILD_DP1K -DCSI_BUILD_REF $(EXTRA_CFLAGS)" \
	CROSS_COMPILE="" NN2_ROOT=${NN2_ROOT} make -C build_script/nn2_dp1k nn2_shared -j8
	cd source/; find . -name *.o | xargs rm; cd -

.PHONY: install_nn2
install_nn2: include
	mkdir -p install_nn2/lib
	cp include install_nn2 -r
	cp lib/libcsi_nn2_* install_nn2/lib -rf


clean:
	rm lib/* -rf
	find . -name *.o | xargs rm -rf
