import argparse
import os

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from optim.build_optim import create_optimizer
from src import models_vit
from src.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from src.lr_generator import LearningRate
from src.metrics import EvalNet, Evaluate, final_test, merge
from src.train_one_step import create_train_one_step
# from src.video_dataset import create_dataset
from src.video_dataset_new import create_dataset
from utils_mae.config_parser import get_config
from utils_mae.helper import cloud_context_init
from utils_mae.logger import get_logger
from utils_mae.monitor import LossMonitor


class NetWithLoss(nn.Cell):
    def __init__(self, net, loss, accum_iter=1):
        super().__init__()
        self.net = net
        self.loss = loss
        self.accum_iter = accum_iter
    
    def construct(self, imgs, labels, index=None):
        logits, labels = self.net(imgs, labels)
        loss = self.loss(logits, labels)
        return loss / self.accum_iter
        
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

    if args.test: 
        args.logger.info("Test...")
        preds_file = os.path.join(args.save_dir, str(local_rank)+'.txt')
        final_test(test_dataset, model, args.test_num_spatial_crops, preds_file)
        if args.local_rank == 0:
            final_top1 ,final_top5 = merge(args.save_dir, device_num, device_id)
            args.logger.info(f'final_top1: {final_top1}, final_top5: {final_top5}')
        exit(0)

    # print("Model = %s" % str(model))
    data_size = train_dataset.get_dataset_size()
    total_batch_size = args.batch_size * args.accum_iter * args.device_num
    num_training_steps_per_epoch = data_size // args.accum_iter
    if args.start_learning_rate == 0.:
        args.start_learning_rate = (args.base_lr * total_batch_size) / 256
    args.end_learning_rate = (args.end_learning_rate * total_batch_size) / 256
    args.logger.info("LR = %.8f" % args.start_learning_rate)

    if args.epochs < args.warmup_epochs:
        lr_schedule = args.start_learning_rate
        optimizer = nn.AdamWeightDecay(
            model.trainable_params(),
            learning_rate=lr_schedule, #args.base_lr, #
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2
        )
    else:
        lr_schedule = LearningRate(args.start_learning_rate, args.end_learning_rate, args.epochs, args.warmup_epochs, data_size)
        optimizer = create_optimizer(args, model, lr_schedule)

    if args.mixup_active:
        # smoothing is handled with mixup label transform
        loss = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0:
        loss = LabelSmoothingCrossEntropy(args.label_smoothing)
    else:
        loss = nn.CrossEntropyLoss()
    net_with_loss = NetWithLoss(model, loss, args.accum_iter)

    eval_loss = nn.CrossEntropyLoss()
    eval_model = EvalNet(model, eval_loss)
    # define model
    if args.device_target in ["CPU"]:
        train_model = nn.TrainOneStepCell(model, optimizer)
    else:
        train_model = create_train_one_step(args, net_with_loss, optimizer, log=args.logger)

    # define callback
    callback = [LossMonitor(accum_iter=args.accum_iter, ifeval=args.eval, log=args.logger)]

    # define ckpt config
    save_ckpt_feq = args.save_ckpt_epochs * args.per_step_size
    if local_rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=save_ckpt_feq, keep_checkpoint_max=100, integrated_save=False)
        ckpoint_cb = ModelCheckpoint(prefix=args.prefix, directory=args.save_dir, config=config_ck)
        callback += [ckpoint_cb, ]
    # define Model and begin training
    model = Model(train_model, loss_fn=None, optimizer=None, eval_network=eval_model, eval_indexes=[0,1,2], metrics={"acc1": nn.TopKCategoricalAccuracy(1), "acc5": nn.TopKCategoricalAccuracy(5)})

    # count = 1
    # while args.eval:
    #     args.logger.info(f"Eval {count}")
    #     result = model.eval(val_dataset, dataset_sink_mode=args.sink_mode)
    #     args.logger.info(result)
    #     count += 1

    if args.eval:
        model.fit(new_epochs, train_dataset, valid_dataset=val_dataset, callbacks=callback, dataset_sink_mode=args.sink_mode, valid_dataset_sink_mode=args.sink_mode, sink_size=args.per_step_size)
    else:
        model.train(new_epochs, train_dataset, callbacks=callback, dataset_sink_mode=args.sink_mode, sink_size=args.per_step_size)
    # profiler.analyse()
    args.logger.info("Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    # parser.add_argument('--config_file', type=str, default="config/jupyter_config/finetune/default_finetune_ViT-B.yaml")
    parser.add_argument('--config_file', type=str, default="default_finetune_ViT-B.yaml")
    args = parser.parse_args()
    args = get_config(args.config_file)

    main(args)


