"""
Localizing Anomalies from Weakly-Labeled Videos作者给出的数据集只有切完的frame
https://github.com/ktr-hubrt/WSAL
为了方便观察，此代码将frame级别的数据集重新组装为视频
"""

import os
import cv2


def images_to_video(folder_path, output_file):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')],
                         key=lambda x: int(os.path.splitext(x)[0]))  # 获取文件夹中所有以.jpg结尾的图片文件并按数字大小排序
    frame_rate = 25  # 视频帧率
    video_size = (720, 300)  # 视频尺寸

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置视频编码器
    video = cv2.VideoWriter(output_file, fourcc, frame_rate, video_size)  # 创建视频写入对象

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)  # 图片文件的完整路径

        frame = cv2.imread(image_path)  # 读取图片帧
        frame = cv2.resize(frame, video_size)  # 调整图片尺寸与视频尺寸一致

        video.write(frame)  # 将帧写入视频

    video.release()  # 释放视频写入对象
    print(output_file+" has been created")

def get_subfolders(directory):  # 获取目录下所有文件夹
    subfolders = []  # 用于保存文件夹的列表
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subfolder_path = os.path.join(root, dir)  # 构建文件夹的完整路径
            subfolders.append(subfolder_path)
        break  # 递归深度为1
    return subfolders

frames_path=r"E:\tmp\TAD\frames"
dirs=get_subfolders(frames_path)

for folder_path in dirs:
    # print(item)
    output_file = folder_path.replace(r"E:\tmp\TAD\frames",r"E:\tmp\TAD\videos")# r"E:\tmp\TAD\code_test\video\01_Accident_001.mp4"  # 输出的视频文件名
    images_to_video(folder_path,output_file)
    # pass