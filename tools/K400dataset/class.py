"""
根据记载着视频名称与对应分类的.csv文件，将k400视频文件分类到对应的文件夹中
"""

import os
from glob import glob
import shutil
from pprint import pp


# 读取记录着所有类别名的文件（类别顺序要对齐）
def readClsTable(clstable_path, output_path, buildClsfolder=True):
    if not os.path.isfile(clstable_path):
        raise FileNotFoundError(f"{clstable_path} does not exist!")

    # 读取类别文件
    dict_list = []
    f_dictk400 = open(clstable_path, "r")  # kinetics classes file
    for line in f_dictk400.readlines():
        line = line.strip()
        line = line.replace('"', '')
        dict_list.append(line)
    f_dictk400.close()
    print(dict_list)
    print(f"length of dict_list:{len(dict_list)}", "\nreadClsTable finish")

    # 创建文件夹
    clsfolder_dict = {}
    for i, cls in enumerate(dict_list):
        clsfolder = os.path.join(output_path, cls)
        clsfolder.replace("\\", '/')
        clsfolder_dict[cls] = clsfolder
        if buildClsfolder:
            if not os.path.exists(clsfolder):
                os.makedirs(clsfolder)

    print(clsfolder_dict)
    return dict_list, clsfolder_dict


def classifyMov(movfolder_path: str,
                datalable,
                dict_list: list,
                clsfolder_dict):
    assert len(dict_list) == len(clsfolder_dict) and (len(dict_list) in (400, 600, 700))  # k400

    # 收录train/val.csv中记录的类别映射
    yid_line_dict = {}
    yid_cls_dict = {}
    video_record_count = 0
    same_mov = 0
    # 逐行读取val.csv中记录的文件，放置到相应的文件夹中
    f_val_cvdf = open(datalable, "r")  # e.g., k400 val.csv
    for line in f_val_cvdf.readlines():
        line = line.strip()
        if len(line) == 0:
            continue
        # line = line.replace(" ","_")
        line = line.replace('"', '')
        cls, yid, st, et, split, _ = line.split(",")
        # 以下避免同一个视频的两个片段分属两个类别（实际测试发现不存在这种离谱情况）
        if yid not in yid_cls_dict:
            yid_line_dict[yid] = cls + "," + yid + "," + str(st) + "," + str(et) + "," + str(split) + "\n"
            yid_cls_dict[yid] = cls
        elif cls != yid_cls_dict[yid]:
            assert KeyError, f"yid:{yid} is {cls},but exist {yid_cls_dict[yid]}"
        else:
            same_mov += 1
        video_record_count += 1
        # print(yid_cls_dict[yid])
        # yid_cls_dict[yid] = line[:line.rfind(",")]
    f_val_cvdf.close()
    print("csv记录的视频", video_record_count, "同视频片段：", same_mov)
    print("视频映射数量", len(yid_cls_dict))

    # 遍历文件，根据上面的映射表将文件分到不同类别的文件夹中
    video_files_count = 0
    video_files = []
    norecord_case = 0
    norecord_list = []
    copy_num = 0
    for file_path in glob(os.path.join(movfolder_path, "**/*"), recursive=True):
        file_path.replace(r"\\", r'/')
        # print(file_path)
        if os.path.isfile(file_path) and file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            filename = os.path.basename(file_path)[:-18]
            if filename in yid_cls_dict:
                cls = yid_cls_dict[filename]
                shutil.copy(file_path, clsfolder_dict[cls])
                # print(f"复制文件{filename} cls：{cls}")
                copy_num += 1
            else:
                norecord_case += 1
                norecord_list.append(file_path)
                print(f"norecord_case:{norecord_case}/{video_files_count} ")
            video_files.append(file_path)
            video_files_count += 1


    print(f"norecord_case/video_files_count:{norecord_case}/{video_files_count}")
    print(video_files[:5])
    print("copy_num:", copy_num)

    with open('norecord_list.txt', 'w') as file:  # 打开output.txt文件，'w'表示写入模式
        file.writelines('\n'.join(norecord_list))




if __name__ == '__main__':
    clstable_path = "cls_table.txt"
    output_path = "val"
    mov_path = "val_complete"
    datatable = "val.csv"
    dict_list, clsfolder_list = readClsTable(clstable_path, output_path)
    classifyMov(mov_path, datatable, dict_list, clsfolder_list)
