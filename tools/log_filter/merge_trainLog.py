"""
读取指定文件夹中的多个日志文件，解析每个日志文件中的训练指标，并计算多次训练的平均训练指标。最终将平均训练指标写入一个txt文件中
"""
import os
from glob import glob
import numpy as np

# 指定要读取的日志文件所在的文件夹路径
logs_dir = r'E:\训练作业输出\6.1  finetune 100B surveillance 401class _clsToken\rawLog'
# logs_dir = r'E:\训练作业输出\5.26 finetune 100B surveillance 401class\rawLog'
# logs_dir = r'E:\训练作业输出\5.22 finetune 100B surveillance\rawLog1'
# logs_dir = r'E:\训练作业输出\4.17finetune800_75epoch（7node）\rawLog'

# 获取文件夹中的所有日志文件名
logs_names = os.listdir(logs_dir)

# 创建要输出结果的txt文件
save_txt = open(logs_dir + '.txt', 'w')

# 创建一个空的字典，用于存储每个日志文件的训练指标
result_dict = {}

for log in logs_names:
    # 拼接日志文件的完整路径
    path = os.path.join(logs_dir, log)
    # 为每个日志文件创建一个空字典，用于存储训练指标
    result_dict[log] = {'acc1': [], 'acc5': [], 'loss': []}
    # 读取日志文件内容
    with open(path, 'r', encoding='utf-8') as f:
        # 逐行解析日志文件内容
        log_info = f.readlines()

        # 初始化变量
        count_num = 0
        acc1, acc5, loss = 0, 0, 0

        for line in log_info:
            # 如果当前行包含 acc1、acc5 和 eval_loss 三个指标，就提取出对应的数值，并将其累加到变量中
            if 'acc1' in line and 'acc5' in line and 'eval_loss' in line:
                acc1_ = float(line.split('\'acc1\': ')[1].split(', \'acc5\'')[0])
                acc5_ = float(line.split('\'acc5\': ')[1].split(', \'eval_loss\'')[0])
                eval_loss_ = float(line.split('\'eval_loss\': ')[1].split('}')[0])

                if count_num % 8 == 0 and count_num != 0:
                    # 如果累加次数达到8次(每节点卡数)，就计算平均值，并将其存储到字典中
                    acc1 = acc1_
                    acc5 = acc5_
                    loss = eval_loss_
                else:
                    acc1 += acc1_
                    acc5 += acc5_
                    loss += eval_loss_

                count_num += 1
                if count_num % 8 == 0 and count_num > 0:
                    result_dict[log]['acc1'].append(acc1 / 8)
                    result_dict[log]['acc5'].append(acc5 / 8)
                    result_dict[log]['loss'].append(loss / 8)


# 将每个日志文件的训练指标存储到列表中，便于后续求平均值
acc1_result,acc5_result,loss_result= [],[],[]

for log in logs_names:
    acc1_result.append(result_dict[log]['acc1'])
    acc5_result.append(result_dict[log]['acc5'])
    loss_result.append(result_dict[log]['loss'])


# 使用 numpy 模块求多次训练的平均训练指标
acc1_result_np = np.array(acc1_result)
acc5_result_np = np.array(acc5_result)
loss_result_np = np.array(loss_result)
print(acc1_result_np.shape)
acc1_mean = np.mean(acc1_result_np, axis=0)
acc5_mean = np.mean(acc5_result_np, axis=0)
loss_mean = np.mean(loss_result_np, axis=0)

# 将平均训练指标写入输出文件中
epoch = 0
for acc1, acc5, loss in zip(acc1_mean, acc5_mean, loss_mean):
    epoch += 1
    save_txt.write(f'epoch:{epoch} acc1:{acc1},acc5:{acc5},eval_loss:{loss}\n')
    print(f'epoch:{epoch} acc1:{acc1},acc5:{acc5},eval_loss:{loss}')
# 关闭输出文件
save_txt.close()
