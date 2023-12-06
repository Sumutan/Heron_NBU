import argparse
import os

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from optim.build_optim import create_optimizer
from src import topk
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

    if args.eval:
        val_dataset = create_dataset(args, mode='val', shuffle=False)
        val_data_size = val_dataset.get_dataset_size()
        args.logger.info("Create val dataset finish, data size:{}".format(val_data_size))

    # import time
    # end_time = time.time()
    # for batch in val_dataset.create_dict_iterator():
    #     # print(batch['video'].shape)
    #     # print("代码段运行时间为：", time.time()-end_time, "秒")
    #     end_time = time.time()
    # data io test
    # count = 0
    # args.logger.info(f"start")
    # for batch in val_dataset.create_dict_iterator():
    #     count += 1
    # args.logger.info(f"batch count: {count}")
    # exit(1)

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

    eval_loss = nn.CrossEntropyLoss()
    eval_model = EvalNet(model, eval_loss)

    model = Model(eval_model, loss_fn=None, optimizer=None, eval_network=eval_model, eval_indexes=[0,1,2],
                  # metrics={"acc1": nn.TopKCategoricalAccuracy(1), "acc5": nn.TopKCategoricalAccuracy(5)})
                  metrics={"acc1": topk.TopKCategoricalAccuracy(1), "acc5": topk.TopKCategoricalAccuracy(5)})

    count = 0
    if args.eval:
        args.logger.info(f"Eval {count}")
        result = model.eval(val_dataset, dataset_sink_mode=args.sink_mode)
        args.logger.info(result)
        count += 1

    # if args.eval:
    #     model.fit(new_epochs, train_dataset, valid_dataset=val_dataset, callbacks=callback, dataset_sink_mode=args.sink_mode, valid_dataset_sink_mode=args.sink_mode, sink_size=args.per_step_size)
    # else:
    #     model.train(new_epochs, train_dataset, callbacks=callback, dataset_sink_mode=args.sink_mode, sink_size=args.per_step_size)
    # profiler.analyse()
    args.logger.info("Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    # parser.add_argument('--config_file', type=str, default="config/jupyter_config/finetune/default_config_ViT-B-eval_ddp.yaml")
    parser.add_argument('--config_file', type=str, default="config/jupyter_config/finetune/finetune_ViT-B-eval.yaml")

    args = parser.parse_args()
    args = get_config(args.config_file)

    main(args)

