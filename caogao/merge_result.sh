#!/bin/bash
#用于整合多机多卡的输出，统计总精度

RANK_SIZE=$1
START_RANK=$2

END_RANK=$(($RANK_SIZE+$START_RANK))

for((i=$START_RANK;i<${END_RANK};i++))
do
    cp device$i/output/$(($i-$START_RANK)).txt output/
done

python ./merge_result.py $RANK_SIZE

