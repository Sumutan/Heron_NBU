# 删除空文件夹
import os

def remove_empty_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        if not dirs and not files:
            os.rmdir(root)

path = "/media/dxm/WD_BLACK/social_surveillance_3"

remove_empty_folders(path)