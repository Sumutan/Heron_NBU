#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh RANK_SIZE"
echo "For example: bash run.sh 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

RANK_SIZE=$1

echo "$RANK_SIZE"

EXEC_PATH=$(pwd)

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_4pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_4pcs.json
    export RANK_SIZE=4
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp -r ./config ./device$i
    cp -r ./src ./device$i
    cp -r ./tools ./device$i
    cp -r ./utils ./device$i
    cp ./*.py ./device$i

    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $DEVICE_ID"
    env > env$i.log
    python ./run_pretrain.py > train.log$i 2>&1 &
    if [ $? -eq 0 ];then
        echo "training success"
    else
        echo "training failed"
        exit 2
    fi
    cd ../
done

# rm -rf device0
# mkdir device0
# cp -r ./config ./device0
# cp -r ./src ./device0
# cp -r ./tools ./device0
# cp -r ./utils ./device0
# cp ./*.py ./device0
# cd ./device0
# export DEVICE_ID=2
# export RANK_ID=2
# echo "start training for device 0"
# env > env0.log
# pytest -s -v ./run_pretrain.py > train.log0 2>&1
# if [ $? -eq 0 ];then
#     echo "training success"
# else
#     echo "training failed"
#     exit 2
# fi
# cd ../

