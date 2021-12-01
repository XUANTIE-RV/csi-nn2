#! /bin/bash
#! /bin/bash
# auto run validation test

echo "start test"

if [ ! -d valid_datas ]; then
    mkdir valid_datas
else
    rm ./valid_datas/*
fi

# build the test cases
make clean
make -f Makefile.860

# generate the test datas
cd valid_datas
DATADIR="../python_ref"
for k in $(ls $DATADIR/*.py)
do
    python3 $k
done

# run the test
QEMU_PATH="qemu-cskyv2 -cpu ck860v"
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
