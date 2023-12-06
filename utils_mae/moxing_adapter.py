# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Moxing adapter for ModelArts"""

import os
import functools
from mindspore import context
from mindspore.profiler import Profiler

_global_sync_count = 0

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def sync_data(from_path, to_path):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    """
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", from_path)
        print("to path: ", to_path)
        mox.file.copy_parallel(from_path, to_path)
        print("===finish data synchronization===")
        try:
            os.mknod(sync_lock)
        except IOError:
            print("Failed to create directory")
        print("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(0.001)

    print("Finish sync data from {} to {}.".format(from_path, to_path))


def wrapped_func(config_name):
    """
        Download data from remote obs to local directory if the first url is remote url and the second one is local path
        Upload data from local directory to remote obs in contrast.
    """
    if not os.path.exists(config_name.output_path):
        os.makedirs(config_name.output_path, exist_ok=True)

    # 下面的代码要注意参数配置
    # if config_name.train_url:
    #     if not os.path.exists(config_name.output_path):
    #         os.makedirs(config_name.output_path)
    #     sync_data(config_name.train_url, config_name.output_path)
    #     print("Workspace downloaded: ", os.listdir(config_name.output_path))

def prepareDataCsv():
    """
    每台机器只有一张卡上运行的代码准备csv文件
    """
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("===finish data synchronization===")
        try:
            root_folder_path = [
                "/home/ma-user/modelarts/inputs/data_dir_0/train",
                "/home/ma-user/modelarts/inputs/data_dir_0/val"
            ]
            for path in root_folder_path:
                buildDataCsv(path)
            os.mknod(sync_lock)
        except IOError:
            print("Failed to create directory")
        print("===save flag===")
    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(0.001)

    print("Finish prepareDataCsv.")

def prepareDataCsv_fromCsv_ddp():
    """
    每台机器只有一张卡上运行的代码准备csv文件
    """
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("===finish data synchronization===")
        try:
            buildDataCsv_fromtxt(root_folder_path='/home/ma-user/modelarts/inputs/data_dir_0/',
                                 traintxt='kinetics400_val_list_videos.txt')
            os.mknod(sync_lock)
        except IOError:
            print("Failed to create directory")
        print("===save flag===")
    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(0.001)

    print("Finish prepareDataCsv.")