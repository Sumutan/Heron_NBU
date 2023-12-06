#!/bin/bash
# applicable to Ascend

MPI_HOME=/usr/local/openmpi
export PATH=${MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export MANPATH=${MPI_HOME}/share/man:$MANPATH

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_with_mpi.sh"
echo "=============================================================================================================="
set -e -x

rm -rf device saved_graph
mkdir device

cp -r ./config ./device
cp -r ./lr ./device
cp -r ./optim ./device
cp -r ./script ./device
cp -r ./src ./device
cp -r ./tools ./device
cp -r ./transforms ./device
cp -r ./utils_mae ./device
cp ./*.py ./device
#cp -r ./*.ckpt ./device

cd ./device

#echo "start training"
#mpirun --allow-run-as-root -n 8 python ./run_pretrain.py
#mpirun -n 8 python ./run_pretrain.py
#mpirun --allow-run-as-root -n 8 python ./run_finetune.py
#mpirun --allow-run-as-root -n 4 python ./run_finetune_eval.py
# mpirun --allow-run-as-root -n 4 python ./run_finetune_test.py
# mpirun --allow-run-as-root -n 8 python ./run_finetune_abn.py

#特征提取
set +e  #命令执行失败继续执行下一条

#用于UCF-Crime/TAD/ShanghaiTech特征提取  #记得进代码改运行配置
mpirun -n 8 python ./extract_n_crop_mpi.py
#mpirun -n 8 python ./extract_n_crop_mpi_defToken.py
#mpirun -n 8 python ./extract_n_crop_mpi_DT_SP.py

#9-5 AISO UCF 1crop
#mpirun -n 8 python ./extract_n_crop_mpi_defToken.py --dataset=UCF-Crime --mask_ratio=0.5 --crop_num=1 \
#--use_ckpt=/home/ma-user/work/ckpt/9-5_9-1_finetune.ckpt \
#--output_dir=/home/ma-user/work/features/9-5_9-1_finetune_AISO_0.5_1crop

#mpirun -n 8 python ./extract_n_crop_mpi.py --dataset=UCF-Crime \
#--use_ckpt=/home/ma-user/work/ckpt/9-1_pretrain_frame_with_depth_add_loss_on_surveillance_20w_change_encoder.ckpt \
#--output_dir=/home/ma-user/work/features/9-1_pretrain
#mpirun -n 8 python ./extract_n_crop_mpi_defToken.py --dataset=UCF-Crime --mask_ratio=0.5 \
#--use_ckpt=/home/ma-user/work/ckpt/9-1_pretrain_frame_with_depth_add_loss_on_surveillance_20w_change_encoder.ckpt \
#--output_dir=/home/ma-user/work/features/9-1_pretrain_AISO_0.5

#用于XD_Violence-videomae 特征提取 (等待运行)
#extract_XD_feature() {
#    export ckptName=9-5_9-1_finetune    # 只需要修改特征名字,输出特征文件名=XD_${ckptName}
#    export ckptPath=/home/ma-user/work/ckpt/${ckptName}.ckpt
#    export featureName=XD_${ckptName}_AISO_0.5
#    export output_dir_train=/home/ma-user/work/features/${featureName}/train
#    export output_dir_test=/home/ma-user/work/features/${featureName}/test
#    mkdir -p  ${output_dir_train}
#    mkdir -p  ${output_dir_test}
#
#    mpirun -n 8 python ./extract_n_crop_mpi_defToken.py --use_ckpt ${ckptPath} --mask_ratio 0.5 --dataset XD_violence \
#    --input_dir /home/ma-user/work/dataset/XD_violence/train/frames \
#    --output_dir ${output_dir_train}
#    mpirun -n 8 python ./extract_n_crop_mpi_defToken.py --use_ckpt ${ckptPath} --mask_ratio 0.5 --dataset XD_violence\
#    --input_dir /home/ma-user/work/dataset/XD_violence/test/frames \
#    --output_dir ${output_dir_test}
#}
#
#extract_XD_feature





