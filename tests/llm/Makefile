x86_ref_llama:
	gcc -c -g llama2.c -I../../include -I../../include/csinn
	gcc -c -g model-f16.c -I../../include -I../../include/csinn
	gcc llama2.o model-f16.o -o llama2.elf  ../../install_nn2/x86/lib/libshl.a -lm -static -fopenmp -g

x86_ref_llama_quantize:
	gcc -c -g llama2_quantize.c -I../../include -I../../include/csinn
	gcc -c -g model-f16.c -I../../include -I../../include/csinn
	g++ llama2_quantize.o model-f16.o -o llama2_quantize.elf  ../../install_nn2/x86/lib/libshl.a -lm -static -fopenmp -g

clean:
	rm -rf *.o *.elf
