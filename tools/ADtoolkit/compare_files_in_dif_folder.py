import os
from tqdm import tqdm
def task_collection_farme(root):
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
    print("root_path:", root)
    for dir in os.listdir(root):
        task_list.append(os.path.join(root, dir))
    return task_list

def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

if __name__ == '__main__':
    dedpth_frame_folder=r"/home/ma-user/work/dataset/ucf_crime/DepthFrames"
    # r"/home/ma-user/work/dataset/ucf_crime/video_depth"

    error_list=[]
    task_list=task_collection_farme(dedpth_frame_folder)
    print(len(task_list))
    for task in tqdm(task_list):
        frame_num_depth=count_files(task)
        frame_num=count_files(task.replace(r"/DepthFrames",r"/frames"))
        if frame_num_depth!=frame_num:
            error_list.append([task,str(frame_num-frame_num_depth)])
            # print(task,"less frame num:",str(frame_num-frame_num_depth))
            tqdm.write(task+"less frame num:"+str(frame_num-frame_num_depth))
    pass

# dir1 = '/path/to/directory1'
# dir2 = '/path/to/directory2'
#
# count1 = count_files(dir1)
# count2 = count_files(dir2)
#
# print(f"The directory '{dir1}' contains {count1} files.")
# print(f"The directory '{dir2}' contains {count2} files.")
# print(f"The difference in file count is {abs(count1 - count2)}.")