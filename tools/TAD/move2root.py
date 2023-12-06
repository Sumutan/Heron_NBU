"""
将所有子文件夹下的图片放置到根目录下
原文件树示例
frames
- 01_Accident_001.mp4
  |- 0
  |  |- 0
  |     |- 0.jpg
  |     |- 1.jpg
  |     |- 2.jpg
  |  |- 1
  |     |- 100.jpg
  |     |- 101.jpg
  |  |- 2
处理后示例
frames
- 01_Accident_001.mp4
  |- 0.jpg
  |- 1.jpg
  |- 2.jpg
"""

import os
import shutil


def extract_images(root_dir):
    image_count = 0  # 图片计数器

    # 遍历根目录下的所有文件和文件夹，取出jpg
    for root, dirs, files in os.walk(root_dir):
        # 遍历文件
        for file in files:
            if file.endswith(".jpg"):
                # 提取图片到根目录
                src_path = os.path.join(root, file)
                dst_path = os.path.join(root_dir, file)
                shutil.move(src_path, dst_path)
                image_count += 1

    # 删除文件夹，仅保留文件
    dirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):  # 判断是否是文件夹
            dirs.append(item_path)

    # 删除子文件夹（从最深层开始）
    for item in dirs:
        shutil.rmtree(item)

    print(f"根目录{root_dir}下的图片数量：{image_count}")


# 使用示例
root_dir = r"E:\tmp\TAD\code_test\01_Accident_001.mp4"  # 根目录路径


def get_subfolders(directory):  # 获取目录下所有文件夹
    subfolders = []  # 用于保存文件夹的列表
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subfolder_path = os.path.join(root, dir)  # 构建文件夹的完整路径
            subfolders.append(subfolder_path)
        break  # 递归深度为1
    return subfolders


if __name__ == '__main__':
    frames_path=r"E:\tmp\TAD\frames"
    dirs=get_subfolders(frames_path)
    for item in dirs:
        # print(item)
        extract_images(item)
