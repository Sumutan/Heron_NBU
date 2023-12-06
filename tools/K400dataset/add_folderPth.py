import os

# 原始的txt文件路径
# input_file = 'E:/tmp/k400_full/kinetics400_train_list_videos.txt'
# input_file = 'E:/tmp/k400_full/kinetics400_val_list_videos.txt'
input_file ='/home/ma-user/work/dataset/k400-full/kinetics400_train_list_videos.txt'
# input_file = '/home/ma-user/work/dataset/k400-full/kinetics400_val_list_videos.txt'

# 新的txt文件路径
# output_file = 'E:/tmp/k400_full/train.csv'
# output_file = 'E:/tmp/k400_full/val.csv'
output_file = '/home/ma-user/work/dataset/k400-full/train.csv'
# output_file = '/home/ma-user/work/dataset/k400-full/val.csv'

# 文件路径前缀
# prefix = 'E:/tmp/k400_full/videos_train'
prefix = '/home/ma-user/work/dataset/k400-full/videos_train'
# prefix = '/home/ma-user/work/dataset/k400-full/videos_val'
# prefix = '/home/ma-user/modelarts/inputs/data_dir_0/videos_train'
# prefix = '/home/ma-user/modelarts/inputs/data_dir_0/videos_val'

acc=0
# 打开原始的txt文件和新的txt文件
with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    # 逐行读取原始文件中的内容
    for line in f_in:
        # 去掉行末的换行符
        line = line.strip().replace(',', ' ')  # txt to csv
        # 将文件路径前缀和行内容拼接在一起，并写入新文件中
        f_out.write(os.path.join(prefix, line).replace('\\', '/') + '\n')
        acc+=1
print(f"处理{acc}条记录")
