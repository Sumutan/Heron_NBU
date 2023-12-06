import os
import zipfile

from tqdm import tqdm


def unzip_to_folder(zip_file_path):
    # 获取zip文件的路径和名称
    folder_path = os.path.splitext(zip_file_path)[0]

    # 创建同名文件夹
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 解压zip文件到同名文件夹
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)

    # print("解压完成！")
    tqdm.write(f"{os.path.basename(folder_path)}解压完成！")

    os.remove(zip_file_path)

def get_zip_files(path):
    """
    根据输入文件夹路径制作待处理文件列表，output_path与cover可以查看已存在的文件，避免重复运行
    :param path: inout path
    :param output_path: 输出文件存放路径
    :param cover: 是否覆盖运行
    :return: 待处理文件路径列表
    """
    video_files = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath) and filename.endswith('.zip'):
            video_files.append(filepath)
        elif os.path.isdir(filepath):
            video_files += get_zip_files(filepath)
    return video_files

if __name__ == '__main__':

    # 假设要解压的zip文件名为example.zip
    folder = '/home/ma-user/work/dataset/XD_violence/train/frames'
    task_list=get_zip_files(folder)

    bar = tqdm(total=len(task_list))
    for zip_file_path in get_zip_files(folder):
        unzip_to_folder(zip_file_path)
        bar.update(1)
    bar.close()  # 记得关闭
