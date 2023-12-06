import os
import cv2
input_folder = "/media/dxm/WD_BLACK/public_surveillance_video"
output_folder = "/media/dxm/WD_BLACK/public_surveillance_video_crop"

fps = 25
size = (565, 320)
# size_ori = (1920, 1080)

# crop_size (469, 266)   [27:293, 48:517]
# 南联村公变北4号 (461, 266) [27:293, 52:513]


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
                                          (498, 255))

            success, _ = videoCapture.read()
            while success:
                success, frame = videoCapture.read()
                try:
                    frame_crop = frame[27:282, 0:498]
                    videoWriter.write(frame_crop)
                except:
                    break
            print("done!")


print("视频处理完成！")