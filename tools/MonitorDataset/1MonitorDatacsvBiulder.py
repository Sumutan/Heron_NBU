import os
import csv


count=0
# 判断一个文件是否是视频文件
def is_video_file(file_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    ext = os.path.splitext(file_path)[1].lower()
    return ext in video_extensions

# 递归遍历目录，将视频文件的绝对路径与类别标号存储到CSV文件中
def traverse_directory(dir_path, csv_writer):
    global count
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory.")
        return

    for entry in os.scandir(dir_path):
        if entry.is_dir():
            traverse_directory(entry.path, csv_writer)  # 递归遍历子目录
        elif is_video_file(entry.path):
            csv_writer.writerow([entry.path, 400])  # 存储视频文件的路径与类别标号
            count+=1

# 测试
if __name__ == '__main__':
    dir_path = '/home/ma-user/work/dataset/k400-full/surveillance_video'
    csv_file_path = '/home/ma-user/work/dataset/k400-full/monitor.csv'

    # 打开CSV文件并写入表头
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file,delimiter=' ')
        # 遍历目录并将视频文件的路径与类别标号存储到CSV文件中
        traverse_directory(dir_path, csv_writer)

    print(f"CSV file saved to {csv_file_path}. count:{count}")


