# -*- coding: gbk -*-
import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

# ��ȡtxt�ļ����ݲ�����
def read_txt_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            video_info = line.strip().split()
            video_name = video_info[0]
            time_intervals = [int(x) for x in video_info[1:]]
            data.append((video_name, time_intervals))
    return data

# ������Ƶ��ǩ
def generate_labels(video_path, time_intervals, total_frames):
    labels = np.zeros(total_frames)
    for i in range(0, len(time_intervals), 2):
        start_frame = time_intervals[i]
        end_frame = time_intervals[i + 1]
        labels[start_frame:end_frame + 1] = 1
    return labels


# �ļ�������Ƶ��·��
video_folder_path = '/home/ma-user/work/dataset/XD_violence/test/videos' #'../test/videos'
# �����txt�ļ�·��
txt_file_path = '/home/ma-user/work/dataset/XD_violence/XD_Annotation.txt'
# ���pickle�ļ�·��
output_pickle_path = '/home/ma-user/work/dataset/XD_violence/XD-violence.pickle'

# ��ȡtxt�ļ�����
data = read_txt_file(txt_file_path)

# �洢��Ƶ���Ͷ�Ӧ��ǩ���ֵ�
video_labels = {}

# ����ÿ����Ƶ��Ϣ
progress_bar = tqdm(total=len(data), unit='task')
for video_name, time_intervals in data:
    video_path = os.path.join(video_folder_path, video_name+'.mp4')

    # ����Ƶ�ļ�
    cap = cv2.VideoCapture(video_path)

    # ��ȡ��֡��
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # �ر���Ƶ�ļ�
    cap.release()

    # ���ɱ�ǩ
    labels = generate_labels(video_path, time_intervals, total_frames)

    # �洢��Ƶ���ͱ�ǩ
    video_labels[video_name] = labels
    progress_bar.update(1)

# ����Ƶ���ͱ�ǩ�洢Ϊpickle�ļ�
with open(output_pickle_path, 'wb') as f:
    pickle.dump(video_labels, f)

print("Data saved to pickle file.")
