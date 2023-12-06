
import csv
import os

"""
input_file:'file.csv'
output_file:'file.csv'/'outfile.csv'
"""
def replacePATH(input_file,output_file):

    # 定义文件路径和标号的列索引
    PATH_COL_IDX = 0
    LABEL_COL_IDX = 1

    # 打开CSV文件
    with open(input_file, newline='') as csvfile:
        # 读取CSV文件
        csvreader = csv.reader(csvfile, delimiter=' ')
        # 创建一个空列表来存储处理后的数据
        processed_data = []
        # 遍历每一行数据
        for row in csvreader:
            # 获取文件路径
            path = row[PATH_COL_IDX]
            # 将文件路径中的'/home/ma-user/work/dataset/k400-full/'替换为'/home/ma-user/modelarts/inputs/data_dir_0/'
            path = path.replace('/home/ma-user/work/dataset/k400-full/', '/home/ma-user/modelarts/inputs/data_dir_0/')
            # path = path.replace('/home/ma-user/modelarts/inputs/data_dir_0/', '/home/ma-user/work/dataset/k400-full/')  #逆向替换
            # 将处理后的数据添加到列表中
            processed_data.append([path, row[LABEL_COL_IDX]])

    # 将处理后的数据写入新的CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        # 创建CSV写入器
        csvwriter = csv.writer(csvfile, delimiter=' ')
        # 写入数据
        csvwriter.writerows(processed_data)

if __name__ == '__main__':
    # 定义要处理的文件名称列表
    INPUT_FILES = ['/home/ma-user/work/dataset/k400-full/k400_withMonitorCsv401_14w/train.csv',
                   '/home/ma-user/work/dataset/k400-full/k400_withMonitorCsv401_14w/val.csv',
                   '/home/ma-user/work/dataset/k400-full/k400_withMonitorCsv401_14w/test.csv']

    OUTPUT_FILE = ['/home/ma-user/work/dataset/k400-full/k400_withMonitorCsv401_14w/train.csv',
                   '/home/ma-user/work/dataset/k400-full/k400_withMonitorCsv401_14w/val.csv',
                   '/home/ma-user/work/dataset/k400-full/k400_withMonitorCsv401_14w/test.csv']

    for input_file, output_file in zip(INPUT_FILES, OUTPUT_FILE):
        print(f'Processing file: {input_file}   Output file: {output_file}')
        replacePATH(input_file, output_file)

