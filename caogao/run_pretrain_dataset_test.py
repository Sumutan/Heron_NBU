"""
仅仅用于测试pratrain阶段Dataloader功能
拖到项目根目录下使用
"""

import argparse
from src.video_dataset_new import create_dataset

from utils_mae.config_parser import get_config
from utils_mae.helper import cloud_context_init
from utils_mae.logger import get_logger


def main(args):
    context_config = {
        "mode": args.mode,
        "device_target": args.device_target,
        "device_id": args.device_id,
        'max_call_depth': args.max_call_depth,
        'save_graphs': args.save_graphs,
    }
    parallel_config = {
        'parallel_mode': args.parallel_mode,
        'gradients_mean': args.gradients_mean,
    }
    local_rank, device_id, device_num = cloud_context_init(seed=args.seed,
                                                           use_parallel=args.use_parallel,
                                                           context_config=context_config,
                                                           parallel_config=parallel_config)
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)
    args.logger.info(f"local_rank: {local_rank}, device_num: {device_num}, device_id: {device_id}")


    # train dataset
    dataset = create_dataset(args, mode="pretrain", shuffle=True)
    # testdataset = create_dataset(args, mode="pretrain_val", shuffle=False)  #报错

    data_size = dataset.get_dataset_size()
    new_epochs = (args.epochs - args.epochstart)
    if args.per_step_size > 0:
        new_epochs = int((data_size / args.per_step_size) * (args.epochs - args.epochstart))
    elif args.per_step_size == 0:
        args.per_step_size = data_size
    args.logger.info("Will be Training epochs:{}, sink_size:{}".format(new_epochs, args.per_step_size))
    args.logger.info("Create training dataset finish, data size:{}".format(data_size))

    data_iter = dataset.create_dict_iterator(num_epochs=1)
    for batch in data_iter:
        print(list(batch.keys()))
        # 在这里处理数据和目标

    args.logger.info("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    # parser.add_argument('--config_file', type=str,
    #                     default="config/jupyter_config/default_videomae_v2_ViT-B.yaml")
    parser.add_argument('--config_file', type=str,
                        default="config/jupyter_config/default_videomae_v2_ViT-B-copy.yaml")
    args = parser.parse_args()
    args = get_config(args.config_file)

    main(args)
