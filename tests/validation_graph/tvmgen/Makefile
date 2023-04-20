CC = riscv64-unknown-linux-gnu-gcc
INCLUDE = -I../../../include -I../../utils
CFLAGS = -O2 -g3 -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -static

test_objs =

test_objs += reg.o
test_objs += callback.o

all: csi

csi: $(test_objs)

$(test_objs): %.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $< -o $@
	$(CC) $@ $(CFLAGS)  -L../../../c906_static_build/ \
	../../../c906_static_build/libshl_c906.a -lc -lm -lstdc++ -o $@.elf

clean:
	rm -rf  $(test_objs) *.a *.asm *.elf *.bin *.asm
