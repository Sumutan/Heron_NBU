import os
import cv2
# input_folder = "/media/lizi/WD_BLACK/Raw_surveillance_video"
# output_folder = "/media/lizi/WD_BLACK/public_surveillance_video"
input_folder = f"J:\Raw_surveillance_video"
output_folder = f"J:\public_surveillance_video"

fps = 25
size = (565, 320)
# size_ori = (1920, 1080)

for folder_1 in os.listdir(input_folder):
    folder_1_path = os.path.join(input_folder, folder_1)

    output_folder_1_path = os.path.join(output_folder, folder_1)
    os.makedirs(output_folder_1_path, exist_ok=True)

    for folder_2 in os.listdir(folder_1_path):
        folder_path = os.path.join(folder_1_path, folder_2)

        output_folder_2_path = os.path.join(output_folder_1_path, folder_2)
        os.makedirs(output_folder_2_path, exist_ok=True)

        for video_file_name in os.listdir(folder_path):

            video_file_path = os.path.join(folder_path, video_file_name)
            # dav_to_h264
            new_name = video_file_path.split('.')[0] + '.h264'
            os.rename(os.path.join(video_file_path), os.path.join(new_name))

            save_name = video_file_name.split('.')[0] + '.mp4'
            save_path = os.path.join(output_folder_2_path, save_name)

            videoCapture = cv2.VideoCapture(new_name)
            videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

            success, _ = videoCapture.read()
            while success:
                success, frame = videoCapture.read()
                try:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    videoWriter.write(frame)
                except:
                    break
            videoWriter.release()
            print("done!")

print("视频处理完成！")
