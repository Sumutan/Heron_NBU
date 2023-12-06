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


def assign_tasks(tasks, processors=8):
    """
    :param processors: int num of processors
    :param tasks: task_list ,such as: [("任务1", 5), ("任务2", 8), ("任务3", 3), ("任务4", 2), ("任务5", 6)]
    :return: processor_tasks, load_difference ,max_load
             各处理器任务列表 ，  最大任务量差异 ， 最大任务量
    """

    # 获取平均分配的情况


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
    # ...


def get_run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="rgb", type=str)
    parser.add_argument('--use_ckpt',
                        default="/home/ma-user/work/ckpt/9-5_9-1_finetune.ckpt",  # 9-5_9-1_finetune.ckpt
                        type=str)

    # ShanghaiTech
    # parser.add_argument('--dataset', default="ShanghaiTech", choices=['UCF-Crime', 'TAD', 'ShanghaiTech',"XD-Violence"], type=str)
    # parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/ShanghaiTech/frames", type=str)
    # parser.add_argument('--output_dir',default="/home/ma-user/work/features/SHT_9-5_9-1_finetune",type=str)
    # UCF-Crime
    parser.add_argument('--dataset', default="UCF-Crime", choices=['UCF-Crime', 'TAD', 'ShanghaiTech', "XD-Violence"],
                        type=str)
    parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/ucf_crime/frames", type=str)
    parser.add_argument('--output_dir',
                        default="/home/ma-user/work/features/10-13_10-4_finetune",
                        type=str)
    # XD-Violence
    # parser.add_argument('--dataset', default="XD-Violence", choices=['UCF-Crime', 'TAD', 'ShanghaiTech',"XD-Violence"], type=str)
    # parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/XD-Violence/train/frames", type=str)
    # parser.add_argument('--output_dir',
    #                     default="/home/ma-user/work/features/XD_9-5_9-1_finetune/train",
    #                     type=str)
    # TAD
    # parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/TAD/frames", type=str)
    # parser.add_argument('--output_dir', default="/home/ma-user/work/features/TAD_9-5_9-1_finetune",  # test
    #                     type=str)

    parser.add_argument('--use_parallel', default=True, type=bool)  # True / False
    parser.add_argument('--crop_num', type=int, default=10, choices=[1, 10])  # 1/10
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_mode', default="oversample", type=str)
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--config_file', type=str,
                        default="config/jupyter_config/finetune/finetune_ViT-B-eval_copy.yaml")

    run_args = parser.parse_args()

    return run_args


if __name__ == '__main__':
    run_args = get_run_args()

    local_rank, device_id, device_num = 0, 0, 4
    print(f"local_rank:{local_rank}, device_id:{device_id},device_num:{device_num}")

    task_list = get_video_list_with_framenum_continue(run_args.input_dir, run_args.output_dir)

    processor_tasks, load_difference, max_load = assign_tasks(task_list, processors=device_num)  # result: 2d task list
    print(f"max load:{max_load},max load diff：{load_difference}")

    # 使用嵌套的列表推导式提取每个元组的第一个元素
    task_list = [[item[0] for item in sublist] for sublist in processor_tasks]
    local_task_list = task_list[local_rank]

    nums = len(local_task_list)
    # dataset = run_args.dataset
    # for inputs in zip(local_task_list, [run_args.output_dir] * nums, [run_args.batch_size] * nums,
    #                   [run_args.crop_num] * nums, [dataset] * nums):
    #     run(inputs)
