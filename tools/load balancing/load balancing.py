import os
import numpy as np


def assign_tasks(tasks, processors=8):
    """
    :param processors: int num of processors
    :param tasks: task_list ,such as: [("任务1", 5), ("任务2", 8), ("任务3", 3), ("任务4", 2), ("任务5", 6)]
    :return: processor_tasks, load_difference
    """
    sorted_tasks = sorted(tasks, key=lambda x: x[1], reverse=True)
    processor_times = [0] * processors
    processor_tasks = [[] for _ in range(processors)]

    for task in sorted_tasks:
        min_time = min(processor_times)
        min_index = processor_times.index(min_time)
        processor_tasks[min_index].append(task)
        processor_times[min_index] += task[1]

    max_load = max(processor_times)
    min_load = min(processor_times)
    load_difference = max_load - min_load

    return processor_tasks, load_difference, max_load


def assign_tasks_avg(tasks, processors=8):
    """
    :param processors: int num of processors
    :param tasks: task_list ,such as: [("任务1", 5), ("任务2", 8), ("任务3", 3), ("任务4", 2), ("任务5", 6)]
    :return: processor_tasks, load_difference
    按照下标直接切分
    """
    processor_times = [0] * processors
    processor_tasks = [[] for _ in range(processors)]

    task_range = np.linspace(0, len(tasks) - 1, processors + 1).astype(int)
    cost_list = tasks[:, 1]
    for i in range(processors):
        processor_times[i] = np.sum(cost_list[task_range[i]:task_range[i + 1]])

    max_load = max(processor_times)
    min_load = min(processor_times)
    load_difference = max_load - min_load

    return processor_tasks, load_difference, max_load


def count_files_in_folder(folder_path):  # 输入一个文件夹路径，返回其中的文件数量
    count = 0
    for _, _, files in os.walk(folder_path):
        count += len(files)
    return count


def get_video_list_with_framenum_all(input_dir):  # input_dir:'/home/ma-user/work/dataset/ucf_crime/frames'
    video_list = []  # 包含所有视频文件的列表
    cost_list = []
    for video in os.listdir(input_dir):
        cost_list.append(count_files_in_folder(os.path.join(input_dir, video)))
        video_list.append(os.path.join(input_dir, video))
    print("get {} videos".format(len(video_list)))
    return zip(video_list, cost_list)


def get_video_list_with_framenum_continue(input_dir, output_dir):  # 执行未处理的任务
    video_list = []  # 包含所有视频文件的列表
    cost_list = []
    for video in os.listdir(input_dir):
        save_file = '{}_{}.npy'.format(video, "videomae")
        if save_file in os.listdir(os.path.join(output_dir)):
            # print("{} has been extracted".format(save_file))
            pass
        else:
            # print(videos)
            video_list.append(os.path.join(input_dir, video))
            cost_list.append(count_files_in_folder(os.path.join(input_dir, video)))
    print("leave {} videos".format(len(video_list)))
    return zip(video_list, cost_list)


if __name__ == '__main__':
    input_dir = "/home/ma-user/work/dataset/ucf_crime/frames"
    output_dir = "/home/ma-user/work/features/finetune-400_frame_with_depth_add_loss_on_surveillance_20w-100_469"
    # task_list = get_video_list_with_framenum_all(input_dir)
    task_list = get_video_list_with_framenum_continue(input_dir, output_dir)

    # task_list = [("任务1", 5), ("任务2", 8), ("任务3", 3), ("任务4", 2), ("任务5", 6)]
    result, load_diff, max_load = assign_tasks(task_list, processors=8)

    # assess
    # for i, processor_tasks in enumerate(result):
    #     print(f"处理器 {i + 1} 的任务列表：")
    #     for task in processor_tasks:
    #         print(task[0], f"{task[1]}s")
    #     print()

    print(f"负载最高的 CPU 和负载最低的 CPU 之间的差距为：{load_diff}")
    print(f"CPU最高负载 ：{max_load}")
