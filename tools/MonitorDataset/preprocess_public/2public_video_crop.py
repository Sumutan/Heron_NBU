"""
用于三级目录的crop（社会场景）
去除视频边缘的字幕信息
"""

import os
import cv2

def count_mp4_files(path):
    count = 0
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath) and filename.endswith('.mp4'):
            count += 1
        elif os.path.isdir(filepath):
            count += count_mp4_files(filepath)
    return count

# input_folder = r"F:\test"
# output_folder = r"F:\test_done"
# input_folder = "../public_surveillance_video_resized"
# output_folder = "../public_surveillance_video_croped"
input_folder = r"F:\public_surveillance_video_resized"
output_folder = r"F:\public_surveillance_video_croped"


fps = 25
size = (565, 320)
# size_ori = (1920, 1080)

# 校园监控crop
# crop_size (469, 266)   [27:293, 48:517]
# 社会监控crop
# crop_size (474, 231)   [25:256, 0:474]


total=count_mp4_files(input_folder)
count=0
for folder_1 in os.listdir(input_folder):
    folder_1_path = os.path.join(input_folder, folder_1)

    output_folder_1_path = os.path.join(output_folder, folder_1)
    os.makedirs(output_folder_1_path, exist_ok=True)

    for folder_2 in os.listdir(folder_1_path):
        folder_path = os.path.join(folder_1_path, folder_2)

        output_folder_2_path = os.path.join(output_folder_1_path, folder_2)
        os.makedirs(output_folder_2_path, exist_ok=True)

        # 对当前文件夹中的每个视频进行处理
        for video_file_name in os.listdir(folder_path):
            video_file_path = os.path.join(folder_path, video_file_name)
            videoCapture = cv2.VideoCapture(video_file_path)

            crop_filename = "crop_{}.mp4".format(os.path.splitext(os.path.basename(video_file_name))[0])
            crop_file_path = os.path.join(output_folder_2_path, crop_filename)

            videoWriter = cv2.VideoWriter(crop_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25.0,
                                          (474, 231))  # 修改1

            success, _ = videoCapture.read()
            while success:
                success, frame = videoCapture.read()
                try:
                    frame_crop = frame[25:256, 0:474]  # h,w  修改2
                    videoWriter.write(frame_crop)
                except:
                    break
            count+=1
            print(f"{count}/{total}")

print("视频处理完成！")
