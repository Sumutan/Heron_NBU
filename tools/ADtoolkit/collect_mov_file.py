import os
def get_video_files(path,output_path=None,cover=True):
    """
    根据输入文件夹路径制作待处理文件列表，output_path与cover可以查看已存在的文件，避免重复运行
    :param path: inout path
    :param output_path: 输出文件存放路径
    :param cover: 是否覆盖运行
    :return: 待处理视频文件路径列表
    """
    video_files = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath) and (filename.endswith('.mp4') or filename.endswith('.avi')):
            if not cover and output_path:
                if os.path.isfile(os.path.join(output_path,filename)):
                    print(f"{os.path.join(output_path,filename)} is existed.")
                    continue
            video_files.append(filepath)
        elif os.path.isdir(filepath):
            video_files += get_video_files(filepath)
    return video_files