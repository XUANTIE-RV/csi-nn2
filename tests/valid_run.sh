#! /bin/bash
# auto run validation test

if [ $# -lt 1 ];then
	echo "Please input test board args, such as: 860/906"
	exit 1
fi

echo "start test"

TEST_TYPE="$1"

if [ ! -d valid_datas ]; then
    mkdir valid_datas
else
    rm ./valid_datas/*
fi

if [ "$TEST_TYPE" = "860" ]; then
    # build the test cases
    make clean
    make -f Makefile.860
    # qemu command
    QEMU_PATH="qemu-cskyv2 -cpu ck860v"
elif [[ "$TEST_TYPE" = "906" ]]; then
    make clean
    make -f Makefile.906
    # qemu command
    QEMU_PATH="qemu-riscv64"
else
    echo "${TEST_TYPE} is not in the support board list"
    exit 1
fi


# generate the test datas
cd valid_datas
DATADIR="../python_ref"
for k in $(ls $DATADIR/*.py)
do
    python $k
done

function qemu_p()
{
    echo "$QEMU_PATH $1 $2 $3"
    $QEMU_PATH $1 $2 $3
    if [ $? -ne 0 ]; then
        echo "FAIL: $QEMU_PATH $1 $2 $3" | tee -a ./result.sum
    else
        echo "PASS: $QEMU_PATH $1 $2 $3" | tee -a ./result.sum
    fi
}

ELFDIR="../validation"
for k in $(ls $ELFDIR/*.elf)
do
    string=${k##*/}
    if [[ $string == *f32* ]]
    then
        tmp=${string%_f32*}
        filename=`ls ${tmp}_data_f32.bin`
        qemu_p $k $filename 0.0001
    elif [[ $string == *u8* ]]
    then
        tmp=${string%_u8*}
        filename=`ls ${tmp}_data_f32.bin`
        qemu_p $k $filename 0.1
    elif [[ $string == *i8* ]]
    then
        tmp=${string%_i8*}
        filename=`ls ${tmp}_data_f32.bin`
        qemu_p $k $filename 0.1
    fi

done
