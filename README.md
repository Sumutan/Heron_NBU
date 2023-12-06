# SVMAE

[论文主要方法图]




### Environment Prepare
install MindSpore

    conda install mindspore=2.2.0 -c mindspore -c conda-forge

or

    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/unified/x86_64/mindspore-2.2.0-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

### Strat Pretrain on MinsSpore Platform
The following is the distributed training startup command, which we typically use for distributed training on Ascend hardware platforms：

    bash run_with_mpi.sh

### Vision Feature extraction
You can extract visual features for downstream anomaly detection network training with the following command：

    bash run_with_mpi_extractor.sh
    

    