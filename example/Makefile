
c906_m1_f16:
	riscv64-unknown-linux-gnu-gcc c906_mobilenetv1_f16.c -o c906_mobilenetv1_f16.elf -I../include ../install_nn2/lib/libshl_c906.a -lm -static

c906_c2d_f32:
	riscv64-unknown-linux-gnu-gcc c906_conv2d_f32.c -o c906_conv2d_f32.elf  -I../include ../install_nn2/lib/libshl_c906.a -lm -static

clean:
	rm -rf *.elf
