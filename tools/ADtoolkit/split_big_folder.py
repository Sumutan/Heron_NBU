import os
import shutil
from tqdm import tqdm

"""
Q:请编写python代码，对于给定两个文件夹路径path1和path2，path1装着若干文件夹，命名如path1/xx_x264，每个文件夹中都包含着大量图片,
图片命名如“img_00000001.jpg”,请将这些装着大量图片的文件夹拆分成小文件夹放在目录path2下，并重命名为“path2_1”"path2_2"等。
每个文件夹下包含连续图片，每个文件夹下图片数量不超过16000张。
"""


def split_images_into_folders(path1, path2, max_images_per_folder=16000):
    # 获取 path1 下的所有目录
    directories = [os.path.join(path1, d) for d in os.listdir(path1)
                   if os.path.isdir(os.path.join(path1, d))]
    directories = [d for d in directories if d.endswith('_x264')]

    # 创建 path2 目录（如果它还不存在）
    os.makedirs(path2, exist_ok=True)

    # 遍历所有目录
    for directory in directories:
        # 初始化文件计数器和目录计数器
        file_counter = 0
        dir_counter = 1
        next_threashoud = max_images_per_folder

        # 获取目录中的所有图片
        images = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
        images_num = len(images)

        folder_base_name = os.path.basename(directory)
        aim_dir = f'{os.path.join(path2, folder_base_name)}_{dir_counter}'
        os.makedirs(aim_dir, exist_ok=True)
        # 遍历所有图片
        for image_id in range(1, images_num + 1):
            # 如果当前目录的图片数量达到了上限，就创建一个新目录
            if file_counter >= next_threashoud:
                print(f"folder {dir_counter} have finished!")
                dir_counter += 1
                next_threashoud += max_images_per_folder

                # 创建新目录（如果它还不存在）
                aim_dir = os.path.join(f'{os.path.join(path2, folder_base_name)}_{dir_counter}')
                os.makedirs(aim_dir, exist_ok=True)

            # 将图片移动到新目录
            img_name = "img_{:08d}.jpg".format(image_id)  # RGB
            # img_name = "img_{:d}.jpg".format(image_id)  # Depth
            shutil.copy(os.path.join(directory, img_name), os.path.join(aim_dir, img_name))
            file_counter += 1


# 使用示例
# path1 = '/home/ma-user/work/dataset/ucf_crime_307/frames'
# path2 = '/home/ma-user/work/dataset/ucf_crime_307_cliped/frames'
# split_images_into_folders(path1, path2)
#
path1 = '/home/ma-user/work/dataset/ucf_crime_308/frames'
path2 = '/home/ma-user/work/dataset/ucf_crime_308_cliped/frames'
split_images_into_folders(path1, path2)

# path1 = '/home/ma-user/work/dataset/ucf_crime_633/frames'
# path2 = '/home/ma-user/work/dataset/ucf_crime_633_cliped/frames'
# split_images_into_folders(path1, path2)


# path1 = '/home/ma-user/work/dataset/ucf_crime_307/DepthFrames'
# path2 = '/home/ma-user/work/dataset/ucf_crime_307_cliped/DepthFrames'
# split_images_into_folders(path1, path2)
#
# path1 = '/home/ma-user/work/dataset/ucf_crime_308/DepthFrames'
# path2 = '/home/ma-user/work/dataset/ucf_crime_308_cliped/DepthFrames'
# split_images_into_folders(path1, path2)

# path1 = '/home/ma-user/work/dataset/ucf_crime_633/DepthFrames'
# path2 = '/home/ma-user/work/dataset/ucf_crime_633_cliped/DepthFrames'
# split_images_into_folders(path1, path2)
