"""
分布式初始化代码实例
modelarts上不需要配置ranktable
"""
import os
import numpy as np
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication import get_group_size, get_rank, init

PARALLEL_MODE = {"DATA_PARALLEL": context.ParallelMode.DATA_PARALLEL,
                 "SEMI_AUTO_PARALLEL": context.ParallelMode.SEMI_AUTO_PARALLEL,
                 "AUTO_PARALLEL": context.ParallelMode.AUTO_PARALLEL,
                 "HYBRID_PARALLEL": context.ParallelMode.HYBRID_PARALLEL}
MODE = {"PYNATIVE_MODE": context.PYNATIVE_MODE,
        "GRAPH_MODE": context.GRAPH_MODE}

def cloud_context_init(
        seed=0,
        use_parallel=True,
        context_config=None,
        parallel_config=None
):
    np.random.seed(seed)
    set_seed(seed)
    device_num = 1
    device_id = 0
    rank_id = 0
    context_config["mode"] = MODE[context_config["mode"]]
    if use_parallel:
        device_id = int(os.getenv('DEVICE_ID'))  # 0 ~ 7
        context_config["device_id"] = device_id
        context.set_context(**context_config)
        init()
        rank_id = get_rank()  # local_rank
        device_num = get_group_size()  # world_size
        parallel_config["parallel_mode"] = PARALLEL_MODE[parallel_config["parallel_mode"]]
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, **parallel_config)
    else:
        context.set_context(**context_config)
    os.environ['MOX_SILENT_MODE'] = '1'
    return rank_id, device_id, device_num
    # 以128卡运行为例子： rank_id:该实例在集群中的编号（0-127）, device_id：在节点中的编号（0-7）, device_num：集群总实例数量：128


def main(args):
    context_config = {
        "mode": args.mode,  # "PYNATIVE_MODE" "GRAPH_MODE"
        "device_target": args.device_target,  # Ascend/GPU/CPU
        "device_id": args.device_id,  #  0
        'max_call_depth': args.max_call_depth, # 10000
        'save_graphs': args.save_graphs,   # False
    }
    parallel_config = {
        'parallel_mode': args.parallel_mode,
        'gradients_mean': args.gradients_mean,
    }
    local_rank, device_id, device_num = cloud_context_init(seed=args.seed,
                                                           use_parallel=args.use_parallel,
                                                           context_config=context_config,
                                                           parallel_config=parallel_config)
    print(f"local_rank: {local_rank}, device_num: {device_num}, device_id: {device_id}")
    args.device_num = device_num
    args.local_rank = local_rank