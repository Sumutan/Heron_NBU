import os

video_extensions = ['.mp4', '.avi', '.mkv'] # 视频文件的扩展名

def count_video_files(path):
    count = 0 # 初始化视频文件数量
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path): # 如果当前项是文件夹
            count += count_video_files(item_path) # 递归统计子文件夹中的视频文件数量
        elif os.path.isfile(item_path) and os.path.splitext(item_path)[1] in video_extensions: # 如果当前项是视频文件
            count += 1
            if count%1000==0:
                print(count)
                print(item_path)
    return count

# folder_path = '/home/ma-user/work/dataset/k400-full/videos_train' # 文件夹路径
# video_count = count_video_files(folder_path) # 统计视频文件数量
# print(f"文件夹 {folder_path} 中共有 {video_count} 个视频文件")

file='/home/ma-user/work/dataset/k400-full/videos_train/AwXZME3U5EM.mp4'

print(os.path.isfile(file))



# import csv
# import os
#
# csv_file = '/home/ma-user/work/dataset/k400-full/train.csv' # csv文件路径
# count = 0 # 初始化存在数据的数量
# line=0
# with open(csv_file, 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         line+=1
#         video_path = row[0] # 视频数据的位置
#         if os.path.exists(video_path): # 如果视频数据存在
#             count += 1
#         if line%1000==0:
#             print(f'{count}/{line}')
#             print(video_path)
#
# print(f"共有 {count} 条数据存在")