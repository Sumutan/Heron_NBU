"""
递归/遍历列表 检查视频数据集完整性：
    文件是否可以成功打开，或是否存在
"""

import os
import cv2
import csv
import numpy as np
import random
import math
from tqdm import tqdm

not_exist_count = 0  # 问题文件
acc = 0  # 总文件


def deep_check_method(video_path):
    def get_start_end_idx(video_size, clip_size, clip_idx, num_clips, use_offset=False):
        delta = max(video_size - clip_size, 0)
        if clip_idx == -1:
            # Random temporal sampling.
            start_idx = random.uniform(0, delta)
        else:
            if use_offset:
                if num_clips == 1:
                    # Take the center clip if num_clips is 1.
                    start_idx = math.floor(delta / 2)
                else:
                    # Uniformly sample the clip with the given index.
                    start_idx = clip_idx * math.floor(delta / (num_clips - 1))
            else:
                # Uniformly sample the clip with the given index.
                start_idx = delta * clip_idx / num_clips
        end_idx = start_idx + clip_size - 1
        return int(start_idx), int(end_idx)

    def temporal_sampling(frames, start_idx, end_idx, num_samples):
        index = np.linspace(start_idx, end_idx, num_samples).astype(int)
        index = np.clip(index, 0, frames.shape[0] - 1).astype(np.long)
        new_frames = frames[index, :, :, :]
        return new_frames

    # -1 indicates random sampling.
    temporal_sample_index = -1
    spatial_sample_index = -1

    # Try to decode and sample a clip from a video. If the video can not be
    # decoded, repeatly find a random video replacement that can be decoded.
    sample = video_path
    cap = cv2.VideoCapture(sample)
    if not cap.isOpened():
        print("视频打开失败,video path: {}".format(sample))
        if not os.path.exists(sample):
            print("文件{}不存在".format(sample))

    total_frames = cap.get(7)  # 视频总帧数

    frames_list = []
    # try selective decoding.
    clip_size = 16 * 4
    start_idx, end_idx = get_start_end_idx(
        total_frames,
        clip_size,
        temporal_sample_index,
        0,
        use_offset=True,
    )

    video_start_frame = int(start_idx)
    video_end_frame = int(end_idx)
    # video_start_frame = int(0) # 前向对齐用
    # video_end_frame = int(63)

    cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

    frames = []
    # 读取指定数量的帧
    for j in range(clip_size):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    frames = np.array(frames)
    num_frames = 16
    frames = temporal_sampling(frames, 0, clip_size - 1, num_frames)

    # print(frames.shape)   #(16, 266, 468, 3)

    cap.release()

    if len(frames.shape) != 4:
        return 1
    else:
        return 0


# 定义递归函数,检查文件是否可以打开
def traverse_folder(folder_path):
    global count, acc
    acc, count = 0, 0
    # 遍历文件夹中的文件
    for file_name in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file_name)

        # 如果是文件夹，则递归遍历子文件夹
        if os.path.isdir(file_path):
            traverse_folder(file_path)

        # 如果是视频文件，则尝试打开并判断能否打开
        elif file_name.endswith('.mp4') or file_name.endswith('.avi') or file_name.endswith('.mov'):
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    count += 1
                    # os.remove(file_path)
                    print(f'{file_name} deleted: cap=False')
                else:
                    cap.release()
            except:
                # os.remove(file_path)
                print(f'{file_name} deleted: except')
                count += 1
        acc += 1
        if acc % 1000 == 0:
            print(acc)
    print("无法打开的文件夹：{}/{}".format(count, acc))


# 打开CSV文件并读取所有行
def check_csv(csvfile='train.csv', delete=False, deepCheck=True):
    open_error = 0
    with open(csvfile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        global acc, count
        acc, count = 0, 0
        videopaths=[]
        for row in reader:
            videopaths.append(row[0].split(' ')[0])
        print(len(videopaths))

        for video_path in tqdm(videopaths):
            acc += 1
            # 获取每行第一个元素（视频文件路径）
            # video_path = row[0].split(' ')[0]
            # 检查视频文件路径是否存在
            if not os.path.exists(video_path):
                count += 1
                print(video_path + " does not exist.")

            # 检查视频文件路径是否能打开
            if deepCheck:
                count += deep_check_method(video_path)
            else:
                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        open_error += 1
                        if delete: os.remove(video_path)
                        print(f'{video_path} deleted: cap=False')
                    else:
                        cap.release()
                except:
                    if delete:  os.remove(video_path)
                    print(f'{video_path} deleted: except')
                    open_error += 1

            if acc % 1000 == 0:
                print(f"{count + open_error}/{acc}")
    print("无法找到的文件夹：{}/{}".format(count, acc))
    print("无法打开的文件夹：{}/{}".format(open_error, acc))
    print("可用文件数：{}/{}".format(acc - count - open_error, acc))


if __name__ == '__main__':
    # case1:递归遍历
    # # 指定要遍历的文件夹路径
    # folder_path = "E:/tmp/k400_full/videos_val"
    # # 调用递归函数遍历文件夹
    # traverse_folder(folder_path)

    # case2：列表遍历
    check_csv('/home/ma-user/work/dataset/k400-full/onlyMonitorCsv_jpt/train.csv')
