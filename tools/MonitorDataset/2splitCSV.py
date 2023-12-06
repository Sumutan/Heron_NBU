"""
将一个csv文件切分为2部分，并分别保存到2个文件夹中
"""

import pandas as pd


csv='/home/ma-user/work/dataset/k400-full/monitor.csv'
csv1='/home/ma-user/work/dataset/k400-full/monitor_train.csv'
csv2='/home/ma-user/work/dataset/k400-full/monitor_val.csv'

# 读取CSV文件
df = pd.read_csv(csv)

# 切分数据
df1 = df.iloc[:135325]  # 0-58263行
df2 = df.iloc[135325:]  # 余下的行

# 将两个数据帧保存为新的CSV文件
df1.to_csv(csv1, index=False)
df2.to_csv(csv2, index=False)