import argparse
import os

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from src import models_vit
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

    # profiler = ms.Profiler(output_path='./profiler_data')

    args.mixup_active = args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None
    if args.cutmix_minmax == 'None':
        args.cutmix_minmax = None

    # train dataset
    train_dataset = create_dataset(args, mode='finetune', shuffle=True)
    data_size = train_dataset.get_dataset_size()

    new_epochs = args.epochs
    if args.per_step_size > 0:
        new_epochs = int((data_size / args.per_step_size) * args.epochs)
    elif args.per_step_size == 0:
        args.per_step_size = data_size
    args.logger.info("Will be Training epochs:{}, sink_size:{}".format(new_epochs, args.per_step_size))
    args.logger.info("Create train dataset finish, data size:{}".format(data_size))

    if args.eval:
        val_dataset = create_dataset(args, mode='val', shuffle=False)
        val_data_size = val_dataset.get_dataset_size()
        args.logger.info("Create val dataset finish, data size:{}".format(val_data_size))
    if args.test:
        test_dataset = create_dataset(args, mode='test', shuffle=False)
        test_data_size = test_dataset.get_dataset_size()
        args.logger.info("Create test dataset finish, data size:{}".format(test_data_size))

    # create model
    args.logger.info("Create model...")
    model = models_vit.VisionTransformer_v2(**vars(args))
    size = ops.Size()
    n_parameters = sum(size(p) for p in model.trainable_params() if p.requires_grad)
    args.logger.info("number of params: {}".format(n_parameters))

    # if args.test:
    #     args.logger.info("Test...")
    #     preds_file = os.path.join(args.save_dir, str(local_rank) + '.txt')
    #     final_test(test_dataset, model, args.test_num_spatial_crops, preds_file)
    #     if args.local_rank == 0:
    #         final_top1, final_top5 = merge(args.save_dir, device_num, device_id)
    #         args.logger.info(f'final_top1: {final_top1}, final_top5: {final_top5}')
    #     exit(0)

    # print("Model = %s" % str(model))
    data_size = train_dataset.get_dataset_size()
    total_batch_size = args.batch_size * args.accum_iter * args.device_num
    num_training_steps_per_epoch = data_size // args.accum_iter
    if args.start_learning_rate == 0.:
        args.start_learning_rate = (args.base_lr * total_batch_size) / 256
    args.end_learning_rate = (args.end_learning_rate * total_batch_size) / 256
    args.logger.info("LR = %.8f" % args.start_learning_rate)

    data_iter =train_dataset.create_dict_iterator(num_epochs=1)
    for batch in data_iter:
        print(list(batch.keys()))
        # 在这里处理数据和目标


    args.logger.info("Finished!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    # parser.add_argument('--config_file', type=str, default="config/cpu/finetune/videomae_v2_ViT-B_codetest.yaml")
    parser.add_argument('--config_file', type=str, default="config/jupyter_config/finetune/default_config_ViT-B.yaml")
    args = parser.parse_args()
    args = get_config(args.config_file)
    main(args)

    # print(os.path.exists(r"4CqGj7P9nhU"))