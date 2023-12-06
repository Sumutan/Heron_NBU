import argparse
import os
import moxing as mox
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
from src.video_dataset import create_dataset
from utils_mae.config_parser import get_config
from utils_mae.helper import cloud_context_init
from utils_mae.logger import get_logger
from utils_mae.monitor import LossMonitor
from tools.K400dataset.KeneteDatacsvBuilder import buildDataCsv

from utils_mae.moxing_adapter import get_device_id, get_device_num
import time
from mindspore import save_checkpoint

_global_sync_count = 0


def sync_data(from_path, to_path):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    """
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", from_path)
        print("to path: ", to_path)
        mox.file.copy_parallel(from_path, to_path)
        print("===finish data synchronization===")
        try:
            os.mknod(sync_lock)
        except IOError:
            print("Failed to create directory")
        print("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(0.001)

    print("Finish sync data from {} to {}.".format(from_path, to_path))



def wrapped_func(config_name):
    """
        Download data from remote obs to local directory if the first url is remote url and the second one is local path
        Upload data from local directory to remote obs in contrast.
    """
    os.makedirs(config_name.output_path,exist_ok=True)



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
        preds_file = os.path.join(args.save_dir, str(local_rank) + '.txt')
        final_test(test_dataset, model, args.test_num_spatial_crops, preds_file)
        if args.local_rank == 0:
            final_top1, final_top5 = merge(args.save_dir, device_num, device_id)
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
            learning_rate=lr_schedule,  # args.base_lr, #
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2
        )
    else:
        lr_schedule = LearningRate(args.start_learning_rate, args.end_learning_rate, args.epochs, args.warmup_epochs,
                                   data_size)
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
    if args.device_target in ["CPU", "GPU"]:
        train_model = nn.TrainOneStepCell(model, optimizer)
    else:
        train_model = create_train_one_step(args, net_with_loss, optimizer, log=args.logger)

    # define callback
    callback = [LossMonitor(accum_iter=args.accum_iter, ifeval=args.eval, log=args.logger)]

    # define ckpt config
    save_ckpt_feq = args.save_ckpt_epochs * args.per_step_size
    if local_rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=save_ckpt_feq, keep_checkpoint_max=100,
                                     integrated_save=False)
        ckpoint_cb = ModelCheckpoint(prefix=args.prefix, directory=args.ckpt_path, config=config_ck)
        callback += [ckpoint_cb, ]

    # define Model and begin training
    model = Model(train_model, loss_fn=None, optimizer=None, eval_network=eval_model, eval_indexes=[0, 1, 2],
                  metrics={"acc1": nn.TopKCategoricalAccuracy(1), "acc5": nn.TopKCategoricalAccuracy(5)})
    # count = 1
    # while args.eval:
    #     args.logger.info(f"Eval {count}")
    #     result = model.eval(val_dataset, dataset_sink_mode=args.sink_mode)
    #     args.logger.info(result)
    #     count += 1

    run_error=False
    try:
        if args.eval:
            model.fit(new_epochs, train_dataset, valid_dataset=val_dataset, callbacks=callback,
                      dataset_sink_mode=args.sink_mode, valid_dataset_sink_mode=args.sink_mode,
                      sink_size=args.per_step_size)
        else:
            model.train(new_epochs, train_dataset, callbacks=callback, dataset_sink_mode=args.sink_mode,
                        sink_size=args.per_step_size)
    except:
        run_error = True
        args.logger.info("训练中断!")
        save_checkpoint(optimizer, '/cache/train/optimizer_ckpt.ckpt')  # args.output_path+ckptname
        save_checkpoint(model.train_network, '/cache/train/model_ckpt.ckpt')

    # profiler.analyse()
    args.logger.info("Finished!")

    for root, dirs, files in os.walk(args.output_path):
        print(root, dirs, files)

    print("args.output_path", args.output_path)
    train_dir = "obs://heron-nbu/output/finetune_128__100e_B_pre800_tube"
    mox.file.copy_parallel(args.output_path, train_dir)  # args.output_path：/cache/train ; train_dir:obs_path
    if run_error:
        mox.file.copy_parallel("/home/ma-user/modelarts/log", train_dir+"/log")    # 保存npu运行日志

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument('--config_file', type=str, default="config/cloud_config/finetune/finetuneNJU_ViT-B_ddp_cloud.yaml")
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--train_url', type=str, default="")
    parser.add_argument("--output_path", type=str, default="/cache/train", help="dir of training output for local")
    args = parser.parse_args()
    wrapped_func(args)
    # prepareDataCsv()
    args = get_config(args.config_file)

    main(args)
