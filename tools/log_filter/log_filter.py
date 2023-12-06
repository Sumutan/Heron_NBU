"""
用于整理训练日志的输出
"""

import re
import os

input_log = 'log/100epoch finetune 训练日志.log'
output_log = 'eval_acc_log.log'  # 中间文件：精简后的日志
save_output_log=True

with open(input_log, 'r', encoding='utf-8') as input_file:
    with open(output_log, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            if line.startswith('Epoch time:'):
                output_file.write(line)

group_size = 8

input_file = output_log
output_file = 'eval_acc_avg.log'

with open(input_file, 'r') as f:
    lines = f.readlines()

groups = [lines[i:i + group_size] for i in range(0, len(lines), group_size)]

with open(output_file, 'w') as f:
    for i, group in enumerate(groups):
        acc1_values = []
        for line in group:
            if line.startswith('Epoch time:'):
                acc1_match = re.search(r"'acc1': (\d+\.\d+)", line)
                if acc1_match:
                    acc1_values.append(float(acc1_match.group(1)))
        if acc1_values:
            avg_acc1 = sum(acc1_values) / len(acc1_values)
            f.write(f"epoch {i + 1}: Average accuracy = {avg_acc1:.4f}\n")

if not save_output_log:
    os.remove(output_log)
