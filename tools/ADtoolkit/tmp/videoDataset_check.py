"""
递归检查视频数据集文件是否可以成功打开
输入文件夹路径如"/home/ma-user/work/dataset/k400-full"，该程序将递归检查文件夹下所有视频文件是否可以被cv2顺利打开
参考数据集文件数量：
K400:
    train:240436
    val:19796
UCF:1900
"""

import os
import cv2
from tqdm import tqdm
from logger import write_list_to_txt, write_list_to_csv


def checkMovFile_and_getFrameNum(video_path):
    """
    打开单个视频文件，获取视频的总帧数(int)，如果读取失败则返回文件路径(str)
    """
    # 检测文件存在
    if not os.path.isfile(video_path):
        print("文件不存在:", video_path)
        return video_path
    # 检查视频是否成功打开
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("无法打开视频文件:", video_path)
        return video_path
    # 获取视频的总帧数
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # 释放视频对象
    video.release()
    return total_frames


def task_collection(folder_path):
    """获取folder_path下所有的视频文件路径"""
    task_list = []
    for root, dirs, files in os.walk(folder_path):
        print("folder_path:", root)
        for file_name in files:
            if file_name.endswith('.mp4') or file_name.endswith('.avi') or file_name.endswith('.mov'):
                task_list.append(os.path.join(root, file_name))
    return task_list

def task_collection_farme(folder_path):
    """
    获取folder_path下所有的视频文件路径
    目录路径：
        root/
         -folder1/
            -1.jpg
            ...
         -folder2/
            -1.jpg
         ...
    """
    task_list = []
    for root, dirs, files in os.walk(folder_path):
        print("folder_path:", root)
        for dir in dirs:
            task_list.append(os.path.join(root, dir))
    return task_list


def check_mov_in_folder_compare(folder_path1,folder_path2):
    """
    递归检查文件夹下所有视频文件是否可以被cv2顺利打开
    """
    # task_collection
    video_path_list1 = task_collection(folder_path1)
    video_path_list2 = task_collection(folder_path2)
    video_path_list1.sort()
    error_list, video_list = [], []  # 异常视频列表,正常视频的路径及帧数
    less_list=[]
    totle_frame_num = 0

    # process
    for video_path in tqdm(video_path_list1):
        # if 'Normal_Videos549_x264' not in video_path: continue
        re = checkMovFile_and_getFrameNum(video_path)
        if isinstance(re, str):  # 如果视频无法打开，则返回文件路径
            error_list.append(re)
        if isinstance(re, int):
            # video_list.append(f"{video_path} {str(re)}")
            video_list.append([video_path, str(re)])
            totle_frame_num += re
        re2 = checkMovFile_and_getFrameNum(video_path.replace(folder_path1,folder_path2))
        print(re)
        print(re2)
        if re != re2:
            less_list.append([video_path.replace(folder_path1,folder_path2), str(re-re2)])
    # save
    if len(less_list) > 0:
        write_list_to_txt(less_list, "/home/ma-user/work/dataset/ucf_crime/less_videos.txt")
    if len(error_list) > 0:
        write_list_to_txt(error_list, "error_videos.txt")
    write_list_to_txt(video_list, "video_list.txt")
    print(f"问题视频/正常视频: {len(error_list)}/{len(video_path_list1)}")
    print(f"正常视频帧数总和: {totle_frame_num}")
    print(f"不对等视频帧数: {len(less_list)}")

    # write_list_to_csv(video_list,"video_list.csv")


def check_frames_in_folder_compare(folder_path1,folder_path2):
    """
    递归检查文件夹下所有视频文件是否可以被cv2顺利打开
    """
    # task_collection
    video_path_list1 = task_collection(folder_path1)
    # video_path_list2 = task_collection(folder_path2)

    error_list, video_list = [], []  # 异常视频列表,正常视频的路径及帧数
    less_list=[]
    totle_frame_num = 0

    # process
    for video_path in tqdm(video_path_list1):
        re = checkMovFile_and_getFrameNum(video_path)
        if isinstance(re, str):  # 如果视频无法打开，则返回文件路径
            error_list.append(re)
        if isinstance(re, int):
            # video_list.append(f"{video_path} {str(re)}")
            video_list.append([video_path, str(re)])
            totle_frame_num += re
        re2 = checkMovFile_and_getFrameNum(video_path.replace(folder_path1,folder_path2))
        if re != re2:
            less_list.append([video_path.replace(folder_path1,folder_path2), str(re-re2)])
    # save
    if len(less_list) > 0:
        write_list_to_txt(less_list, "/home/ma-user/work/dataset/ucf_crime/less_videos.txt")
    if len(error_list) > 0:
        write_list_to_txt(error_list, "error_videos.txt")
    write_list_to_txt(video_list, "video_list.txt")
    print(f"问题视频/正常视频: {len(error_list)}/{len(video_path_list1)}")
    print(f"正常视频帧数总和: {totle_frame_num}")
    # write_list_to_csv(video_list,"video_list.csv")


# 定义递归函数
def check_mov_in_folder(folder_path):
    """
    递归检查文件夹下所有视频文件是否可以被cv2顺利打开
    """
    # task_collection
    video_path_list = task_collection(folder_path)
    error_list, video_list = [], []  # 异常视频列表,正常视频的路径及帧数
    totle_frame_num = 0

    # process
    for video_path in tqdm(video_path_list):
        re = checkMovFile_and_getFrameNum(video_path)
        if isinstance(re, str):  # 如果视频无法打开，则返回文件路径
            error_list.append(re)
        if isinstance(re, int):
            # video_list.append(f"{video_path} {str(re)}")
            video_list.append([video_path, str(re)])
            totle_frame_num += re
    # save
    if len(error_list) > 0:
        write_list_to_txt(error_list, "error_videos.txt")
    write_list_to_txt(video_list, "video_list.txt")
    print(f"问题视频/正常视频: {len(error_list)}/{len(video_path_list)}")
    print(f"正常视频帧数总和: {totle_frame_num}")
    # write_list_to_csv(video_list,"video_list.csv")


if __name__ == '__main__':
    # check_mov_in_folder(r"/home/ma-user/work/dataset/ucf_crime/video_depth")
    check_mov_in_folder_compare(r"/home/ma-user/work/dataset/ucf_crime/video_depth",r"/home/ma-user/work/dataset/ucf_crime/video_all")