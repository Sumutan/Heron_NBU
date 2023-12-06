import pandas as pd

# 读取两个csv文件
#train
# df1 = pd.read_csv('/home/ma-user/work/dataset/k400-full/train.csv', header=None)
# df2 = pd.read_csv('/home/ma-user/work/dataset/k400-full/monitor_train.csv', header=None)
#val
df1 = pd.read_csv('/home/ma-user/work/dataset/k400-full/val.csv', header=None)
df2 = pd.read_csv('/home/ma-user/work/dataset/k400-full/monitor_val.csv', header=None)

print(df1.columns,df1.shape)
print(df2.columns,df2.shape)

# 合并两个数据框
merged_df = pd.concat([df1, df2], ignore_index=True)

print(merged_df.shape)

# 输出合并后的数据框行数
print('合并后的文件有', len(merged_df), '行')

# 将合并后的数据框保存为一个新的csv文件
# merged_df.to_csv('/home/ma-user/work/dataset/k400-full/k400_withMonitorCsv401_14w/train.csv', index=False, header=None)
# merged_df.to_csv('/home/ma-user/work/dataset/k400-full/k400_withMonitorCsv401_14w/val.csv', index=False, header=None)
merged_df.to_csv('/home/ma-user/work/dataset/k400-full/k400_withMonitorCsv401_14w/test.csv', index=False, header=None)



