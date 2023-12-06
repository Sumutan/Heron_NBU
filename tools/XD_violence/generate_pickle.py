# -*- coding: gbk -*-
import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

# 读取txt文件内容并处理
def read_txt_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            video_info = line.strip().split()
            video_name = video_info[0]
            time_intervals = [int(x) for x in video_info[1:]]
            data.append((video_name, time_intervals))
    return data

# 生成视频标签
def generate_labels(video_path, time_intervals, total_frames):
    labels = np.zeros(total_frames)
    for i in range(0, len(time_intervals), 2):
        start_frame = time_intervals[i]
        end_frame = time_intervals[i + 1]
        labels[start_frame:end_frame + 1] = 1
    return labels


# 文件夹中视频的路径
video_folder_path = '/home/ma-user/work/dataset/XD_violence/test/videos' #'../test/videos'
# 输入的txt文件路径
txt_file_path = '/home/ma-user/work/dataset/XD_violence/XD_Annotation.txt'
# 输出pickle文件路径
output_pickle_path = '/home/ma-user/work/dataset/XD_violence/XD-violence.pickle'

# 读取txt文件内容
data = read_txt_file(txt_file_path)

# 存储视频名和对应标签的字典
video_labels = {}

# 遍历每个视频信息
progress_bar = tqdm(total=len(data), unit='task')
for video_name, time_intervals in data:
    video_path = os.path.join(video_folder_path, video_name+'.mp4')

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 关闭视频文件
    cap.release()

    # 生成标签
    labels = generate_labels(video_path, time_intervals, total_frames)

    # 存储视频名和标签
    video_labels[video_name] = labels
    progress_bar.update(1)

# 将视频名和标签存储为pickle文件
with open(output_pickle_path, 'wb') as f:
    pickle.dump(video_labels, f)

print("Data saved to pickle file.")
