import argparse

from src.video_dataset import create_dataset
from utils_mae.config_parser import get_config
from utils_mae.logger import get_logger
from utils_mae.helper import cloud_context_init


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
    print(f"local_rank: {local_rank}, device_num: {device_num}, device_id: {device_id}")
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)

    args.mixup_active = args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None
    if args.cutmix_minmax == 'None':
        args.cutmix_minmax = None

    # train dataset
    train_dataset = create_dataset(args, mode='finetune', shuffle=True)
    data_size = train_dataset.get_dataset_size()

    data_iter = train_dataset.create_dict_iterator(num_epochs=1)
    for batch in data_iter:
        print(list(batch.keys()))
        break
        # 在这里处理数据和目标

    args.logger.info("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    # parser.add_argument('--config_file', type=str, default="config/cpu/finetune/videomae_v2_ViT-B_codetest.yaml")
    parser.add_argument('--config_file', type=str, default="default_finetune_ViT-B.yaml")
    args = parser.parse_args()
    args = get_config(args.config_file)
    main(args)

    # print(os.path.exists(r"4CqGj7P9nhU"))
