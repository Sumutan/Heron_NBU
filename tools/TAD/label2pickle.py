import pickle


def process_exception_data(file_path):
    exception_data = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip().split()
            file_name = line[0]
            frame_count = int(line[1])
            exceptions = line[2:]

            # 初始化数值列表为全0
            frame_labels = [0] * frame_count

            # 根据异常帧的起始和结束位置将对应的帧标记为异常（值为1）
            for i in range(0, len(exceptions), 2):
                start_frame = int(exceptions[i]) - 1  # 0 is first index in list
                end_frame = int(exceptions[i + 1]) - 1

                if start_frame != -1 and end_frame != -1:
                    for frame in range(start_frame, end_frame + 1):
                        try:
                            frame_labels[frame] = 1
                        except:
                            pass

            exception_data[file_name] = frame_labels

    return exception_data


# 使用示例
file_path = r"E:\tmp\TAD\label\vals.txt"  # 异常定位数据文件路径

exception_data = process_exception_data(file_path)

output_file = "vals.pickle"  # 输出的.pickle文件路径

# 将字典数据保存为.pickle文件
with open(output_file, 'wb') as file:
    pickle.dump(exception_data, file)

print("异常定位数据已保存到pickle文件。")
