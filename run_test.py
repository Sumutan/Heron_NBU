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
# from src import models_mae_v2_view
from src.models_mae_v2_pred import PretrainVisionTransformer as VideoMAEV2


def numpy_array_to_video(numpy_array,video_out_path,fps=25):
    """
    numpy_array:[t,h,w,c]
    """
    video_height = numpy_array.shape[1]
    video_width = numpy_array.shape[2]

    out_video_size = (video_width,video_height)
    output_video_fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))
    video_write_capture = cv2.VideoWriter(video_out_path, output_video_fourcc, fps, out_video_size)

    for frame in numpy_array:
        video_write_capture.write(frame)

    video_write_capture.release()

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
    local_rank, device_id, device_num = cloud_context_init(seed=args.seed,
                                                           use_parallel=args.use_parallel,
                                                           context_config=context_config,
                                                           parallel_config=parallel_config)
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)

    dataset = create_dataset(args, shuffle=True)

    model = VideoMAEV2(**vars(args))

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
    scale = 1.0 / 255.0
    for data in dataset.create_tuple_iterator():
        video, mask, ids_restore, ids_keep, ids_pred = data
        # loss = model(video, mask, ids_restore, ids_keep, ids_pred)
        loss, raw_imgs, masked_imgs, pred_video = model(video, mask, ids_restore, ids_keep, ids_pred)
        print(loss)

        # raw_imgs = video
        if len(raw_imgs.shape) == 4:  # 走这条分支
            b, l, h, w = raw_imgs.shape
            raw_imgs = raw_imgs.reshape(b, 3, args.num_frames, -1, h, w)
            raw_imgs = raw_imgs.transpose(0, 3, 1, 2, 4, 5)  # b, r, c, t, h, w
        if len(raw_imgs.shape) == 6:
            b, r, c, t, h, w = raw_imgs.shape
            raw_imgs = raw_imgs.reshape(b * r, c, t, h, w)  # b*r, c, t, h, w

        b, c, t, h, w = raw_imgs.shape
        raw_imgs = raw_imgs.asnumpy().transpose(0, 2, 3, 4, 1)  # b*r, t, h, w ,c
        for i, imgs in enumerate(raw_imgs):  # 遍历batch
            output_video=[]
            img_ = np.zeros([h, w * 16, c])
            for j, img in enumerate(imgs):  # 遍历t：16 frames
                img = (img * std + mean) / scale
                img_[:, w * j:w * (j + 1), :] = img
                # img_[:, :, :] = img
                output_video.append(img)
            # img_ = img_
            os.makedirs('./out/', exist_ok=True)
            cv2.imwrite(f"./out/raw_{i}.jpg", img_[:, :, ::-1])
            output_video=np.array(output_video, dtype=np.uint8)[:,:,:,::-1]
            numpy_array_to_video(output_video, f"./out/raw_{i}.mp4",fps=4)
            print(f"./out/raw_{i}.mp4")
            # exit()

        # masked_imgs = masked_imgs.asnumpy().transpose(0, 2, 3, 4, 1)
        # for i, imgs in enumerate(masked_imgs):
        #     img_ = np.zeros([h, w*8, c])
        #     for j, img in enumerate(imgs):
        #         img = (img * std + mean) / scale
        #         img_[:, w*j:w*(j+1), :] = img
        #     img_ = img_[:, :, ::-1]
        #     cv2.imwrite(f"masked_{i}.jpg", img_)

        # pred_video = pred_video.asnumpy().transpose(0, 2, 3, 4, 1)
        # for i, imgs in enumerate(pred_video):
        #     img_ = np.zeros([h, w*8, c])
        #     for j, img in enumerate(imgs):
        #         img = (img * std + mean) / scale
        #         img_[:, w*j:w*(j+1), :] = img
        #     img_ = img_[:, :, ::-1]
        #     cv2.imwrite(f"pred_{i}.jpg", img_)

        pred_video = pred_video.asnumpy().transpose(0, 2, 3, 4, 1)  # b*r, t, h, w ,c
        for i, imgs in enumerate(pred_video):  # 遍历batch
            output_video=[]
            img_ = np.zeros([h, w * 16, c])
            for j, img in enumerate(imgs):  # 遍历t：16 frames
                img = (img * std + mean) / scale
                img_[:, w * j:w * (j + 1), :] = img
                output_video.append(img)
            os.makedirs('./out/', exist_ok=True)
            cv2.imwrite(f"./out/pred_{i}.jpg", img_[:, :, ::-1])
            output_video=np.array(output_video, dtype=np.uint8)[:,:,:,::-1]
            numpy_array_to_video(output_video, f"./out/pred_{i}.mp4",fps=4)
            print(f"./out/pred_{i}.mp4")
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    # parser.add_argument('--config_file', type=str,
    #                     default="config/jupyter_config/default_videomae_v2_ViT-B.yaml")
    parser.add_argument('--config_file', type=str,
                        default="config/jupyter_config/videomae_v2_ViT-B_run_test_dev.yaml")

    args = parser.parse_args()
    args = get_config(args.config_file)

    main(args)
