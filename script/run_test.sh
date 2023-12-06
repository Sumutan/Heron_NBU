RANK_SIZE=$1
START_RANK=$2
END_RANK=$(($RANK_SIZE+$START_RANK))
echo "$START_RANK"
echo "$END_RANK"


for((i=$START_RANK;i<${END_RANK};i++))
do
    DEVICE_ID=$i
    echo "start training for device $DEVICE_ID"
done