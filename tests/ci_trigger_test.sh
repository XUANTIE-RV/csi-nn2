#! /bin/bash
# auto run validation test

function qemu_p()
{
    $QEMU_PATH $1 $2 $3
    if [ $? -ne 0 ]; then
        echo "FAIL: $QEMU_PATH $1 $2 $3" | tee ./result.sum
    else
        echo "PASS: $QEMU_PATH $1 $2 $3" | tee ./result.sum
    fi 
}


if [ -d "./tests/valid_datas" ]; then
	rm -rf ./tests/valid_datas
fi

for i in {"860","906"}
do
    
    if [[ $i = "860" ]];then
        TARGET=c$i
        QEMU_PATH="qemu-cskyv2 -cpu ck860v"
    elif [[ $i = "906" ]];then
        TARGET=c$i
        QEMU_PATH="qemu-riscv64"
    fi

    make clean
    make nn2_$TARGET

    pushd ./tests

    if [ ! -d valid_datas ]; then
        mkdir valid_datas
    fi
    echo "Test target $i"

    # build the test cases
    make -f Makefile.$i clean
    make -f Makefile.$i

    # generate the test datas
    cd valid_datas
    python3 ../python_ref/convolution_nchw.py

    # run the test
    qemu_p ../validation/convolution_nchw_f32.o.elf ./convolution_nchw_data_f32.bin 0.0001

    cd ..
    popd

done


TEST=$(cat ./tests/valid_datas/result.sum|grep "FAIL")
if [[ $TEST != "" ]];then
    RET=1
else
    RET=0 
    rm -rf ./tests/valid_datas
fi
exit $RET

    
