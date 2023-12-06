import cv2
import os
from tqdm import tqdm

def extract_motion_scenes(video_file, output_dir, threshold=10, min_scene_duration=1.0, interception_length=10):
    def extract_frames(input_file, output_file, start_frame, end_frame):
        """
        从一段视频中截取指定帧率的视频片段并将其保存为另一个文件。

        Args:
            input_file (str): 要截取的输入视频文件名。
            output_file (str): 要保存的输出视频文件名。
            start_frame (int): 要截取的视频片段的起始帧。
            end_frame (int): 要截取的视频片段的结束帧。

        Returns:
            None
        """
        # 打开输入视频文件
        cap = cv2.VideoCapture(input_file)

        # 获取视频的FPS（每秒帧数）和总帧数
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 确保开始帧数不超过视频的总帧数
        if start_frame > total_frames:
            start_frame = total_frames

        # 设置读取到指定帧之前的帧都舍弃
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 创建视频编写器，设置输出文件的编码和帧率
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # print(f"save size:{output_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4)))}")

        out = cv2.VideoWriter(output_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        # 逐帧读取视频，直到到达结束帧
        count = 0
        while cap.isOpened() and count < end_frame - start_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # 将帧写入输出视频文件
            out.write(frame)
            count += 1

        # 释放视频捕获器和视频编写器
        cap.release()
        out.release()

        print(f'已从视频 {input_file} 中截取 {start_frame / fps} 到 {end_frame / fps} s并保存为 {output_file}。')

    # 视频中有min_scene_duration秒内容连续变化则截取
    # interception_length:截取动态视频段长度（s）
    # 创建输出目录
    output_dir = os.path.splitext(output_dir)[0]
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_file)

    # 获取视频的FPS和帧数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化变量
    prev_frame = None
    scene_start = None
    scene_num = 0
    handel_start = 0

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret: break
        if i < handel_start: continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray_frame
            continue

        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        num_white_pixels = cv2.countNonZero(thresh)

        if num_white_pixels > threshold and scene_start is None:
            scene_start = i

        if scene_start is not None and (num_white_pixels <= threshold or i == frame_count - 1):
            scene_duration = (i - scene_start) / fps
            if scene_duration >= min_scene_duration:
                scene_num += 1
                scene_file = os.path.join(output_dir, f"scene_{scene_num:03d}.mp4")
                if scene_start + fps * interception_length < frame_count:
                    # write_list.append([video_file, scene_file, scene_start, scene_start+fps * interception_length, fps])
                    extract_frames(video_file, scene_file, scene_start, scene_start + fps * interception_length)
                    handel_start = scene_start + fps * interception_length
            scene_start = None

        prev_frame = gray_frame

    cap.release()

#取得文件夹下所有视频文件路径
def get_video_files(path):
    video_files = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath) and (filename.endswith('.mp4') or filename.endswith('.avi')):
            video_files.append(filepath)
        elif os.path.isdir(filepath):
            video_files += get_video_files(filepath)
    return video_files


if __name__ == "__main__":
    video_folder = "/media/dxm/WD_BLACK/resize"
    output_folder = "/media/dxm/WD_BLACK/extract"

    file_split = []
    video_files=get_video_files(video_folder)

    # for file in video_files:
    #     file = os.path.splitext(file)[0]
    #     file_split.append(file)

    video_files_output=[]
    for file in video_files:
        video_files_output.append(file.replace(video_folder,output_folder))

    for video_file,output_file in tqdm(list(zip(video_files,video_files_output))):
        extract_motion_scenes(video_file, output_file)

