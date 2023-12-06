"""
将视频分割成小段，同时resize
实现了多进程版本
"""
import multiprocessing
import os
import cv2
from tqdm import tqdm


def multiprocess_init(use_cores=1024):
    cores = min(use_cores, multiprocessing.cpu_count())
    print("cores：", cores)
    pool = multiprocessing.Pool(processes=cores)
    return pool, cores


def get_video_files(path):  # 获取路径下的所有视频文件的绝对路径列表
    video_files = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath) and (filename.endswith('.mp4') or filename.endswith('.avi')):
            video_files.append(filepath)
        elif os.path.isdir(filepath):
            video_files += get_video_files(filepath)
    return video_files


def video_snipping(video_file, output_dir, frame=64, resize=None):
    """
    将视频video_file切分成frame帧长的片段，存放在output_dir
    :param video_file: 待切分视频的绝对路径
    :param output_dir: 裁切片段的输出目录
    :param frame:  片段长度
    :return:None
    """
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取视频
    cap = cv2.VideoCapture(video_file)

    # 获取视频帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算每个片段的帧数
    num_frames_per_segment = frame  # default:64
    num_segments = (total_frames + num_frames_per_segment - 1) // num_frames_per_segment  # 会保存最后不足64的片段

    # 逐个保存视频片段
    for i in range(num_segments):
        # 读取视频片段
        frames = []
        for j in range(num_frames_per_segment):
            ret, frame = cap.read()
            if not ret:
                break
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frames.append(frame)

        # 保存视频片段
        segment_name = os.path.splitext(os.path.basename(video_file))[0] + f"_{i + 1}.mp4"
        segment_path = os.path.join(output_dir, segment_name)

        height, width, _ = frames[0].shape
        if resize:
            width, height = resize

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)
        writer.release()

    # 释放资源
    cap.release()


def video_snipping_multiprocess(input, frame=64, resize=(565, 320)):
    """
    功能：
        将视频video_file
        resize成resize参数指定大小
        切分成frame帧长的片段
        存放在output_dir下的同名文件夹下
    （多进程版本，区别在输入参数形式不同，这里将多参数压包为input传入）

    :param input:包括
        :param video_file: 待切分视频的绝对路径
        :param output_dir: 裁切片段的输出目录
    :param frame:  片段长度
    :param resize: 裁切画面的大小
    :return:None
    """
    video_file, output_dir = input

    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取视频
    cap = cv2.VideoCapture(video_file)

    # 获取视频帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算每个片段的帧数
    num_frames_per_segment = frame  # default:64
    num_segments = (total_frames + num_frames_per_segment - 1) // num_frames_per_segment  # 会保存最后不足64的片段

    # 逐个保存视频片段
    for i in range(num_segments):
        # 读取视频片段
        frames = []
        for j in range(num_frames_per_segment):
            ret, frame = cap.read()
            if not ret:  # ret 图像是否成功读取
                break
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frames.append(frame)

        # 保存视频片段
        segment_name = os.path.splitext(os.path.basename(video_file))[0] + f"_{i + 1}.mp4"
        segment_path = os.path.join(output_dir, segment_name)

        height, width, _ = frames[0].shape
        if resize:
            width, height = resize

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)
        writer.release()

    # 释放资源
    cap.release()


if __name__ == '__main__':
    # video_folder = r"E:\tmp\NBUabnormal"
    # output_folder = r"E:\tmp\NBUabnormal_out"
    video_folder = r"/media/lizi/My Passport/NBUabnormal"
    output_folder = r"/media/lizi/My Passport/NBUabnormal_snippets_test"


    video_files = get_video_files(video_folder)
    total = len(video_files)

    video_files_output = []
    # for file in video_files:   #输出为文件
    #     video_files_output.append(file.replace(video_folder, output_folder))
    for file in video_files:  # 输出为文件夹
        output_path = os.path.splitext(file.replace(video_folder, output_folder))[0]  # 获取不包含后缀的文件名作为文件夹名
        video_files_output.append(output_path)

    ds = list(zip(video_files, video_files_output))  # 待处理数据列表

    # 单进程处理方式（调试）
    for video_file, output_file in tqdm(ds):
        video_snipping(video_file, output_file, resize=(565, 320))

    # 多进程处理方式
    # pool, cores = multiprocess_init()
    # with tqdm(total=total) as pbar:
    #     for _ in pool.imap_unordered(video_snipping_multiprocess, ds):  # pool.map(f,ds)
    #         pbar.update(1)
#7:32 4线程1h
