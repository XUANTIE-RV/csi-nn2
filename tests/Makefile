TEST_ROOT := $(shell pwd)

all: test_ref test_anole

test_ref_x86:
	make -C validation -f Makefile.ref_x86

test_ref:
	make -C validation -f Makefile.ref

test_c860:
	make -C validation -f Makefile.c860

test_c906:
	make -C validation -f Makefile.c906

test_anole:
	make -C validation_graph -f Makefile.anole

test_pnna:
	make -C validation_graph -f Makefile.pnna

test_pnna_x86:
	make -C validation_graph -f Makefile.pnna_x86


clean:
	rm -rf  *.a *.asm utils/*.o
	cd validation/; 	 find . -name "*.o" -or -name "*.elf" | xargs rm; cd -
	cd validation_graph; find . -name "*.o" -or -name "*.elf" | xargs rm; cd -