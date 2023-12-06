import multiprocessing
import os


def f(x):
    return x * x


cores = multiprocessing.cpu_count()
print("cores：", cores)

pool = multiprocessing.Pool(processes=cores)
xs = range(5)

# # method 1: map
# print pool.map(f, xs)  # prints [0, 1, 4, 9,   16]
#
# # method 2: imap
# for y in pool.imap(f, xs):
#     print y            # 0, 1, 4, 9, 16, respectively
#
# method 3: imap_unordered


def get_video_files(path):  # 获取路径下的所有视频文件的绝对路径列表
    video_files = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath) and (filename.endswith('.mp4') or filename.endswith('.avi')):
            video_files.append(filepath)
        elif os.path.isdir(filepath):
            video_files += get_video_files(filepath)
    return video_files


if __name__ == '__main__':
    video_folder = r"E:\tmp\NBUabnormal"
    output_folder = r"E:\tmp\NBUabnormal_out"

    file_split = []
    video_files = get_video_files(video_folder)

    pass
    # for file in video_files:
    #     file = os.path.splitext(file)[0]
    #     file_split.append(file)

    video_files_output = []
    for file in video_files:
        video_files_output.append(file.replace(video_folder, output_folder))

    # for video_file, output_file in tqdm(list(zip(video_files, video_files_output))):
    #     extract_motion_scenes(video_file, output_file)
