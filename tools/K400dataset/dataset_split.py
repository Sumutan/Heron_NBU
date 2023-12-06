import os
import shutil
import time

# 读取kinetics400_val_list_videos.txt文件中的记录，返回一个列表或字典
def read_val_list(val_list_file, use_dict=False):
    val_list = {}
    with open(val_list_file, 'r') as f:
        for line in f:
            name, category = line.strip().split(' ')
            if use_dict:
                val_list[name] = category
            else:
                val_list.append(name)
    return val_list if use_dict else val_list

# 统计文件夹中的视频文件数量
def count_files(video_folder):
    count = 0
    for filename in os.listdir(video_folder):
        if filename.endswith('.mp4'):  # 假设视频文件的扩展名为.mp4
            count += 1
    return count

# 将文件分成n份并复制到对应的文件夹中
def split_files(val_list, video_folder, output_folder, n):
    video_count = count_files(video_folder)
    each_part = video_count // n
    file_list = [] #根据顺序记录文件名
    print("file_list length:",len(file_list))
    for name in val_list:
        if os.path.exists(os.path.join(video_folder, name)):
            file_list.append(name)
    for i in range(n):
        print(f"copy prat{i+1}")
        start_index = i * each_part
        end_index = (i + 1) * each_part
        if i == n-1:
            end_index = video_count
        part_files = file_list[start_index:end_index]
        part_folder = os.path.join(output_folder, 'part_{}'.format(i+1))
        if not os.path.exists(part_folder):
            os.makedirs(part_folder)
        for filename in part_files:
            src_file = os.path.join(video_folder, filename)
            dst_file = os.path.join(part_folder, filename)
            shutil.copyfile(src_file, dst_file)

# 主函数
def main():
    # val_list_file = "E:/tmp/k400_full/kinetics400_val_list_videos.txt"  # 文件名与类别号的记录文件
    # video_folder = "E:/tmp/k400_full/videos_val"  # 存放视频文件的文件夹
    # output_folder = "E:/tmp/k400_full/videos_val_partial"  # 输出文件夹
    val_list_file = "F:/k400/tar0000/Kinetics-400/kinetics400_train_list_videos.txt"  # 文件名与类别号的记录文件
    video_folder = "F:/k400/tar0000/Kinetics-400/videos_train"   # 存放视频文件的文件夹
    output_folder = "F:/k400/tar0000/Kinetics-400/videos_train_partial"   # 输出文件夹

    n = 4  # 将文件分成n份
    val_list = read_val_list(val_list_file, use_dict=True)
    start_time = time.time()  # 开始计时
    split_files(val_list, video_folder, output_folder, n)
    end_time = time.time()  # 结束计时
    elapsed_time = end_time - start_time  # 计算执行时间
    print('Split {} files into {} parts in {:.2f} seconds'.format(len(val_list), n, elapsed_time))

    acc=0
    for i in range(1,5):
        print(os.path.join(r'E:\tmp\k400_full\videos_val_partial',f'part_{i}'))
        acc+=count_files(os.path.join(r'E:\tmp\k400_full\videos_val_partial',f'part_{i}'))
        print(acc)

if __name__ == '__main__':
    main()