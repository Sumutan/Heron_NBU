import argparse

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.lr_generator import LearningRate
from src.models_mae_v2 import PretrainVisionTransformer as VideoMAEV2
from src.train_one_step import create_train_one_step
# from src.video_dataset import create_dataset
from src.video_dataset_new import create_dataset
from src.metrics import PretrainEvaluate

from utils_mae.config_parser import get_config
from utils_mae.helper import cloud_context_init
from utils_mae.logger import get_logger
from utils_mae.monitor import LossMonitorMAE,LossMonitorMAE_test


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

    # profiler = ms.Profiler(output_path='./profiler_data')

    # train dataset
    dataset = create_dataset(args, mode="pretrain", shuffle=True)
    # testdataset = create_dataset(args, mode="pretrain_val", shuffle=False)

    data_size = dataset.get_dataset_size()
    new_epochs = (args.epochs - args.epochstart)
    if args.per_step_size > 0:
        new_epochs = int((data_size / args.per_step_size) * (args.epochs - args.epochstart))
    elif args.per_step_size == 0:
        args.per_step_size = data_size
    args.logger.info("Will be Training epochs:{}, sink_size:{}".format(new_epochs, args.per_step_size))
    args.logger.info("Create training dataset finish, data size:{}".format(data_size))

    model = VideoMAEV2(**vars(args))
    size = ops.Size()
    n_parameters = sum(size(p) for p in model.trainable_params() if p.requires_grad)
    args.logger.info("number of params: {}".format(n_parameters))

    # print("Model = %s" % str(model))

    if args.start_learning_rate == 0.:
        args.start_learning_rate = (args.base_lr * args.device_num * args.batch_size) / 256
    # define lr_schedule
    if args.epochs < args.warmup_epochs:
        lr_schedule = args.start_learning_rate
    else:
        lr_schedule = LearningRate(args.start_learning_rate, args.end_learning_rate, args.epochs, args.warmup_epochs,
                                   data_size)

    # define optimizer
    optimizer = nn.AdamWeightDecay(
        model.trainable_params(),
        learning_rate=lr_schedule,  # args.base_lr
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2
    )

    # load pretrain ckpt
    if args.use_ckpt:
        args.logger.info(f"Load ckpt: {args.use_ckpt}...")
        params_dict = load_checkpoint(args.use_ckpt)
        msg = load_param_into_net(model, params_dict)
        if len(msg):
            args.logger.info(msg)
        else:
            args.logger.info("All keys match successfully!")
        # load_param_into_net(optimizer, params_dict)

    # define model
    if args.device_target in ["CPU"]:
        train_model = nn.TrainOneStepCell(model, optimizer)
    else:
        train_model = create_train_one_step(args, model, optimizer, log=args.logger)

    # define callback
    callback = [LossMonitorMAE(log=args.logger),]
    # define ckpt config
    save_ckpt_feq = args.save_ckpt_epochs * args.per_step_size
    if local_rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=save_ckpt_feq, keep_checkpoint_max=30, integrated_save=False,
                                     async_save=True, exception_save=True)  # 添加异步保存和中断保存
        ckpoint_cb = ModelCheckpoint(prefix=args.prefix, directory=args.save_dir, config=config_ck)
        callback += [ckpoint_cb]
    print("callback", callback)

    # define Model and begin training
    args.logger.info("Training begin...")
    model = Model(train_model, loss_fn=None, optimizer=None, eval_network=train_model, eval_indexes=[0,0,0], metrics={"rebuild loss": PretrainEvaluate(log=args.logger)})
    if args.eval:
        # model.eval(dataset,callbacks=callback)        #测试eval过程
        model.fit(new_epochs, train_dataset=dataset, valid_dataset=testdataset, valid_frequency=1, callbacks=callback,
                  dataset_sink_mode=args.sink_mode,valid_dataset_sink_mode=False)
    else:
        model.train(new_epochs, dataset, callbacks=callback,
                    dataset_sink_mode=args.sink_mode,)  # , sink_size=args.per_step_size
    # profiler.analyse()
    args.logger.info("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    # parser.add_argument('--config_file', type=str,
    #                     default="config/jupyter_config/default_videomae_v2_ViT-B.yaml")
    # parser.add_argument('--config_file', type=str,
    #                     default="config/jupyter_config/default_videomae_v2_ViT-B-copy.yaml")
    parser.add_argument('--config_file', type=str,
                        default="config/jupyter_config/default_videomae_v2_ViT-L.yaml")

    args = parser.parse_args()
    args = get_config(args.config_file)

    main(args)
