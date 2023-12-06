import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_accuracy(log_file,log_type):
    if log_type=='NJU_log':
        with open(log_file, 'r') as f:
            lines = f.readlines()
        accuracy_list = []
        for line in lines:
            match = re.search(r'"val_acc1": ([\d\.]+)', line)
            if match:
                accuracy_list.append(float(match.group(1)))
        return accuracy_list
    # elif log_type=='ms_log':
    #     with open(log_file, 'r') as f:
    #         lines = f.readlines()
    #     accuracy_list = []
    #     for line in lines:
    #         match = re.search(r'Average accuracy = ([\d\.]+)', line)
    #         if match:
    #             accuracy_list.append(float(match.group(1)))
    #     for i in range(len(accuracy_list)):
    #         accuracy_list[i] = accuracy_list[i] * 100
    elif log_type == 'ms_log':
        with open(log_file, 'r') as f:
            lines = f.readlines()
        accuracy_list = []
        for line in lines:
            match = re.search(r'acc1:([\d\.]+),', line)
            if match:
                accuracy_list.append(float(match.group(1)))
        for i in range(len(accuracy_list)):
            accuracy_list[i] = accuracy_list[i] * 100

        return accuracy_list
    else:
        raise ValueError("error input:log_type")

# log1_accuracy = extract_accuracy('NJUvideoMAE_finetune100_log.txt','NJU_log')
# log2_accuracy = extract_accuracy('4_30_finetune100e_B_RandomMask.txt','ms_log')
# log3_accuracy = extract_accuracy('BlockMask_with_SurveillanceVideo.txt','ms_log')
log4_accuracy = extract_accuracy('finetune 100B surveillance 401class.txt','ms_log')
log5_accuracy = extract_accuracy('6.1  finetune 100B surveillance 401class _clsToken.txt','ms_log')

df = pd.DataFrame({
    # 'NJU Acc': log1_accuracy,
    # 'ms random mask Acc': log2_accuracy,
    # 'BlockMask_Svl': log3_accuracy,
    'BlockMask_Svl_401clsFinetune': log4_accuracy,
    'BlockMask_Svl_401clsFinetune_clsToken': log5_accuracy

})

print(df)

#绘图
# plt.plot(df['NJU Acc'], label='NJU Acc')
# plt.plot(df['ms random mask Acc'], label='ms random mask Acc')
# plt.plot(df['BlockMask_Svl'], label='BlockMask_Svl')
plt.plot(df['BlockMask_Svl_401clsFinetune'], label='BlockMask_Svl_401clsFinetune')
plt.plot(df['BlockMask_Svl_401clsFinetune_clsToken'], label='BlockMask_Svl_401clsFinetune_clsToken')

plt.title('Accuracy Comparison')
plt.xlabel('finetune Epoch')
plt.ylabel('Accuracy')

plt.legend()

plt.show()