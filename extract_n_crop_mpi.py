"""
分布式并行提取特征
输入视频已经切为视频帧形式：
frames_folder/
    videoname1/
        xx0001.jpg
        xx0002.jpg
        ...
    videoname2/
        ...
    ...
"""

import os
from tqdm import tqdm
import argparse
from PIL import Image
import math
from utils_mae.config_parser import get_config
from utils_mae.helper import cloud_context_init
from utils_mae.logger import get_logger
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src import models_vit_feature
import mindspore.ops as ops
import numpy as np
from mindspore.common.tensor import Tensor
import cv2


def downloadmodel(use_ckpt, use_parallel, args):
    """
    :param use_ckpt: 等待提取的模型路径
    :param useParallel: 是否并行
    :return: 模型与运行环境参数
    """
    # parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    # 读入模型生成默认配置
    # parser.add_argument('--config_file', type=str,
    #                     default="config/jupyter_config/finetune/finetune_ViT-B-eval_copy.yaml")
    # args = parser.parse_args()
    # args = parser.parse_known_args()
    args = get_config(args.config_file)

    # 以下修改实现了直接在get_run_args直接修改特征提取的所有参数，不再需要修改yaml文件
    if use_ckpt is not None:
        args.use_ckpt = use_ckpt
    if use_parallel is not None:
        args.use_parallel = use_parallel

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
    # create model
    args.logger.info("Create model...")
    model = models_vit_feature.VisionTransformer_v2(**vars(args))
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
    model.set_train(False)
    return model, local_rank, device_id, device_num


def load_frame(frame_file):
    data = Image.open(frame_file)
    data = data.resize((340, 256), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert (data.max() <= 1.0) and (data.min() >= -1.0)

    return data  # return a frame(np.array)


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (256, 340, 3))
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))

    return batch_data  # shape:(batch_size,num_frames,256, 340, 3) 我猜的


def oversample_data(data: np.ndarray):  # (39, 16, 224, 224, 2)  # 10 crop
    ###---------------I3D model to extract snippet feature---------------------
    # Input:  bx3x16x224x224  np.ndarray
    # Output: bx1024  np.ndarray
    data_flip = data[:, :, :, ::-1, :]

    data_1 = data[:, :, :224, :224, :]
    data_2 = data[:, :, :224, -224:, :]
    data_3 = data[:, :, 16:240, 58:282, :]
    data_4 = data[:, :, -224:, :224, :]
    data_5 = data[:, :, -224:, -224:, :]

    data_f_1 = data_flip[:, :, :224, :224, :]
    data_f_2 = data_flip[:, :, :224, -224:, :]
    data_f_3 = data_flip[:, :, 16:240, 58:282, :]
    data_f_4 = data_flip[:, :, -224:, :224, :]
    data_f_5 = data_flip[:, :, -224:, -224:, :]

    return [
        data_1, data_2, data_3, data_4, data_5,
        data_f_1, data_f_2, data_f_3, data_f_4, data_f_5
    ]


def oversample_data_1crop(video_data, target_size=(224, 224, 3)):  # (39, 16, 224, 224, 3)
    assert target_size[0] == target_size[1]  # 目前之设计了针对正方形裁切
    # 获取输入数据的维度
    B, T, h, w, c = video_data.shape

    # 计算裁切的起始位置
    crop_size = min(h, w)
    if h > w:
        start_h = (h - crop_size) // 2
        start_w = 0
    else:
        start_h = 0
        start_w = (w - crop_size) // 2

    # 裁切视频数据
    cropped_data = video_data[:, :, start_h:start_h + crop_size, start_w:start_w + crop_size, :]  # (B, T, 224, 224, 3)

    # 创建用于存储调整大小后的视频数据的数组
    resized_data = np.zeros((B * T, target_size[0], target_size[1], target_size[2]))
    cropped_data = np.resize(cropped_data, (B * T, crop_size, crop_size, 3))

    # resize每一帧
    for t in range(cropped_data.shape[0]):  # B * T
        frame = cropped_data[t]  # 获取当前帧
        resized_frame = cv2.resize(frame, (target_size[0], target_size[1]))  # 调整当前帧大小 ->(224,224,3)
        resized_data[t] = resized_frame  # 存储调整大小后的帧

    resized_data = np.resize(resized_data, (B, T, target_size[0], target_size[1], target_size[2]))
    # resized_data = np.resize(cropped_data, (B, T, crop_size, crop_size, target_size[2]))
    return [resized_data]


def forward_batch(b_data, net):
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = Tensor(b_data)  # b,c,t,h,w  # bx3x16x224x224
    b_features = net(b_data, None)
    b_features = b_features.asnumpy()
    return b_features


def assign_tasks(tasks, processors=8):
    """
    :param processors: int num of processors
    :param tasks: task_list ,such as: [("任务1", 5), ("任务2", 8), ("任务3", 3), ("任务4", 2), ("任务5", 6)]
    :return: processor_tasks, load_difference ,max_load
             各处理器任务列表 ，  最大任务量差异 ， 最大任务量
    """
    sorted_tasks = sorted(tasks, key=lambda x: x[1], reverse=True)  # reverse=True 推理， False调试
    processor_times = [0] * processors
    processor_tasks = [[] for _ in range(processors)]

    for task in sorted_tasks:
        min_time = min(processor_times)
        min_index = processor_times.index(min_time)
        processor_tasks[min_index].append(task)
        processor_times[min_index] += task[1]

    max_load = max(processor_times)
    min_load = min(processor_times)
    load_difference = max_load - min_load

    return processor_tasks, load_difference, max_load


def count_files_in_folder(folder_path):  # 输入一个文件夹路径，返回其中的文件数量
    count = 0
    for _, _, files in os.walk(folder_path):
        count += len(files)
    return count


def get_video_list_with_framenum_all(input_dir):  # input_dir:'/home/ma-user/work/dataset/ucf_crime/frames'
    video_list = []  # 包含所有视频文件的列表
    cost_list = []
    for video in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video)
        cost_list.append(count_files_in_folder(video_path))
        video_list.append(video_path)
    print("get {} videos".format(len(video_list)))
    return zip(video_list, cost_list)


def get_video_list_with_framenum_continue(input_dir, output_dir):  # 搜集未处理的任务
    video_list = []  # 包含所有视频文件的列表
    cost_list = []
    for video in os.listdir(input_dir):
        save_file = '{}_{}.npy'.format(video, "videomae")
        if save_file in os.listdir(os.path.join(output_dir)):
            # print("{} has been extracted".format(save_file))
            pass
        else:
            video_path = os.path.join(input_dir, video)
            cost_list.append(count_files_in_folder(video_path))
            video_list.append(video_path)
    print("leave {} videos".format(len(video_list)))
    return zip(video_list, cost_list)  # 任务列表与预算


def run(args_item):
    video_dir, output_dir, batch_size, crop_num, dataset = args_item
    chunk_size = 16
    frequency = 16
    video_name = video_dir.split("/")[-1]
    save_file = '{}_{}.npy'.format(video_name, "videomae")
    if save_file in os.listdir(os.path.join(output_dir)):
        print("{} has been extracted".format(save_file))
        return

    # set up the model
    rgb_files = [i for i in os.listdir(video_dir) if i.endswith('jpg')]
    if dataset == "UCF-Crime":
        rgb_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))  # ucf
    elif dataset == "TAD":
        rgb_files.sort(key=lambda x: int(x.split(".")[0]))  # TAD
    frame_cnt = len(rgb_files)

    assert (frame_cnt > chunk_size), print("{} has not enough frames".format(video_dir))

    # 对不能取整的最后一块以最后一帧填充
    # the number of chunnks that can be formed from the images in the folder.
    clipped_cnt = math.ceil(frame_cnt / chunk_size)
    # If there are not enough images, copy the last frames to fill the gap.
    copy_length = (clipped_cnt * frequency) - frame_cnt
    if copy_length != 0:
        copy_imgs = [rgb_files[frame_cnt - 1]] * copy_length
        rgb_files = rgb_files + copy_imgs

    frame_indices = []  # Frames to chunks,contain lists of indices of each chunk,such as:[[0,1,...15][16,...,31]...]
    for i in range(0, clipped_cnt):
        frame_indices.append(
            [j for j in range(i * frequency, i * frequency + chunk_size)])
    frame_indices = np.array(frame_indices)
    assert frame_indices.shape[0] == clipped_cnt

    batch_num = int(np.ceil(clipped_cnt / batch_size))
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)

    full_features = [[] for i in range(crop_num)]

    for batch_id in tqdm(range(batch_num)):
        batch_data = load_rgb_batch(video_dir, rgb_files, frame_indices[batch_id])

        batch_data_n_crop = []
        if crop_num == 1:
            batch_data_n_crop = oversample_data_1crop(batch_data)
        elif crop_num == 10:
            batch_data_n_crop = oversample_data(batch_data)

        for i in range(crop_num):
            assert (batch_data_n_crop[i].shape[-2] == 224)
            assert (batch_data_n_crop[i].shape[-3] == 224)
            full_features[i].append(forward_batch(batch_data_n_crop[i], model))

    full_features = [np.concatenate(i, axis=0) for i in full_features]
    full_features = [np.expand_dims(i, axis=0) for i in full_features]
    full_features = np.concatenate(full_features, axis=0)  # 合并crop
    np.save(os.path.join(output_dir, save_file), full_features)

    print('{} done: {} / {}, {}'.format(
        video_name, frame_cnt, clipped_cnt, full_features.shape))


def get_run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="rgb", type=str)
    parser.add_argument('--use_ckpt',
                        default="/home/ma-user/work/ckpt/extracted/9-1_pretrain_frame_with_depth_add_loss_on_surveillance_20w_change_encoder.ckpt",  # 9-5_9-1_finetune.ckpt/11-1_10-15_finetune_L.ckpt
                        type=str)
        # finetune_pretrain_with_RandomMaskinK400-100_457.ckpt
        # 8-12_finetune-8-10_8-8_k400_with_surveillancel-100_469.ckpt
        # 8-17_8-15_finetune_random_on_surveillance_20w_k400.ckpt

        # finetune-400_frame_with_depth_add_loss_on_surveillance_20w.ckpt
        # 9-1_pretrain_frame_with_depth_add_loss_on_surveillance_20w_change_encoder.ckpt  (waiting)


    # ShanghaiTech
    # parser.add_argument('--dataset', default="ShanghaiTech", choices=['UCF-Crime', 'TAD', 'ShanghaiTech',"XD-Violence"], type=str)
    # parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/ShanghaiTech/frames", type=str)
    # parser.add_argument('--output_dir',default="/home/ma-user/work/features/SHT_9-5_9-1_finetune",type=str)
    # UCF-Crime
    # parser.add_argument('--dataset', default="UCF-Crime", choices=['UCF-Crime', 'TAD', 'ShanghaiTech',"XD-Violence"], type=str)
    # parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/ucf_crime/frames", type=str)
    # parser.add_argument('--output_dir',
    #                     default="/home/ma-user/work/features/10-13_10-4_finetune",
    #                     type=str)
    # XD-Violence
    # parser.add_argument('--dataset', default="XD-Violence", choices=['UCF-Crime', 'TAD', 'ShanghaiTech',"XD-Violence"], type=str)
    # parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/XD-Violence/train/frames", type=str)
    # parser.add_argument('--output_dir',
    #                     default="/home/ma-user/work/features/XD_9-5_9-1_finetune/train",
    #                     type=str)
    # TAD
    parser.add_argument('--dataset', default="TAD", choices=['UCF-Crime', 'TAD', 'ShanghaiTech',"XD-Violence"], type=str)
    parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/TAD/frames", type=str)
    parser.add_argument('--output_dir', default="/home/ma-user/work/features/9-1_pretrain.ckpt",  # test
                        type=str)

    parser.add_argument('--use_parallel', default=True, type=bool)  # True / False
    parser.add_argument('--crop_num', type=int, default=10, choices=[1, 10])  # 1/10
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_mode', default="oversample", type=str)
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--config_file', type=str,
                        default="config/jupyter_config/finetune/finetune_ViT-B-eval_copy.yaml") # finetune_ViT-L-eval.yaml / finetune_ViT-B-eval_copy.yaml

    run_args = parser.parse_args()

    return run_args


if __name__ == '__main__':
    run_args = get_run_args()
    os.makedirs(run_args.output_dir, exist_ok=True)

    model, local_rank, device_id, device_num = downloadmodel(use_ckpt=run_args.use_ckpt,
                                                             use_parallel=run_args.use_parallel, args=run_args)
    print(f"local_rank:{local_rank}, device_id:{device_id},device_num:{device_num}")

    task_list = get_video_list_with_framenum_continue(run_args.input_dir, run_args.output_dir)

    processor_tasks, load_difference, max_load = assign_tasks(task_list, processors=device_num)  # result: 2d task list
    print(f"max load:{max_load},max load diff：{load_difference}")

    # 使用嵌套的列表推导式提取每个元组
    # 的第一个元素
    task_list = [[item[0] for item in sublist] for sublist in processor_tasks]
    local_task_list = task_list[local_rank]

    nums = len(local_task_list)
    dataset = run_args.dataset
    for inputs in zip(local_task_list, [run_args.output_dir] * nums, [run_args.batch_size] * nums,
                      [run_args.crop_num] * nums, [dataset] * nums):
        run(inputs)
