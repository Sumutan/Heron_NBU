import os
import cv2
import argparse

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net

import numpy as np

from utils_mae.logger import get_logger
from utils_mae.helper import cloud_context_init
from utils_mae.config_parser import get_config

from src.video_dataset import create_dataset
from src import models_mae_pred

def main(args):
    # train dataset
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
    local_rank, device_num = cloud_context_init(seed=args.seed,
                                                use_parallel=args.use_parallel,
                                                context_config=context_config,
                                                parallel_config=parallel_config)
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)

    dataset = create_dataset(args, shuffle=True)

    model = models_mae_pred.MaskedAutoencoderViT(**vars(args))

    print("Model = %s" % str(model))

    # load pretrain ckpt
    try:
        params_dict = load_checkpoint(args.use_ckpt)
        msg = load_param_into_net(model, params_dict)
        if len(msg):
            print(msg)
        else:
            print("All keys match successfully!")
            # load_param_into_net(optimizer, params_dict)
    except:
        print("No ckpt used!")

    model.set_train(False)
    mean = 0.45
    std = 0.225
    scale = 1.0/255.0
    for data in dataset.create_tuple_iterator():
        video, mask, ids_restore, ids_keep, ids_pred = data
        loss, raw_imgs, masked_imgs, pred_video = model(video, mask, ids_restore, ids_keep, ids_pred)

        b, r, c, t, h, w = video.shape
        raw_imgs = raw_imgs.asnumpy().transpose(0, 2, 3, 4, 1)
        for i, imgs in enumerate(raw_imgs):
            img_ = np.zeros([h, w*8, c])
            for j, img in enumerate(imgs):
                img = (img * std + mean) / scale
                img_[:, w*j:w*(j+1), :] = img
            img_ = img_[:, :, ::-1]
            cv2.imwrite(f"raw_{i}.jpg", img_)

        masked_imgs = masked_imgs.asnumpy().transpose(0, 2, 3, 4, 1)
        for i, imgs in enumerate(masked_imgs):
            img_ = np.zeros([h, w*8, c])
            for j, img in enumerate(imgs):
                img = (img * std + mean) / scale
                img_[:, w*j:w*(j+1), :] = img
            img_ = img_[:, :, ::-1]
            cv2.imwrite(f"masked_{i}.jpg", img_)

        pred_video = pred_video.asnumpy().transpose(0, 2, 3, 4, 1)
        for i, imgs in enumerate(pred_video):
            img_ = np.zeros([h, w*8, c])
            for j, img in enumerate(imgs):
                img = (img * std + mean) / scale
                img_[:, w*j:w*(j+1), :] = img
            img_ = img_[:, :, ::-1]
            cv2.imwrite(f"pred_{i}.jpg", img_)

        print(loss)
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument('--config_file', type=str, default="config/default_config_ViT-L.yaml")
    args = parser.parse_args()
    args = get_config(args.config_file)

    main(args)


