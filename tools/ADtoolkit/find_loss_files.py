"""
Q：给定两个文件夹A和B的路径，文件夹B中的文件是文件夹A的子集，请编写程序找出文件夹B相对于文件夹A缺少的文件，存放在列表中
"""

import os
import csv


def write_list_to_txt(lst, file_path='output.txt'):
    """输入一个列表或string，将列表中元素逐行写入文件中"""
    if type(lst) not in [str, list]:
        raise TypeError("lst must be a list or string")
    elif type(lst) == str:
        lst = [lst]
    with open(file_path, "w") as file:
        for item in lst:
            file.write(str(item) + "\n")


# 指定文件夹A和B的路径
folder_a = '/home/ma-user/work/dataset/ucf_crime/video_all'
folder_b = '/home/ma-user/work/dataset/ucf_crime/video_depth'

# 获取文件夹A和B中的文件列表
files_in_a = set(os.listdir(folder_a))
files_in_b = set(os.listdir(folder_b))

# 使用集合差集操作找出文件夹B相对于文件夹A缺少的文件
missing_files = files_in_a - files_in_b

# 将结果转换为列表
missing_files_list = list(missing_files)

print(missing_files_list)
print(len(missing_files_list))

write_list_to_txt(missing_files_list, "/home/ma-user/work/dataset/ucf_crime/require_ucf_depth.txt")