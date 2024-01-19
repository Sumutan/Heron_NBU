"""
extract_n_crop_mpi_DifToken_StructPooling.py
分布式并行提取特征
输入视频已经切为视频帧形式：
frames_folder/
    videoname1/
        xx0001.jpg
        xx0002.jpg
        ...
    videoname2/
        ...
    ...
"""

import os
import time
from tqdm import tqdm
import argparse
from PIL import Image
import math
from utils_mae.config_parser import get_config
from utils_mae.helper import cloud_context_init
from utils_mae.logger import get_logger
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter
from src import models_vit_feature
# from src import models_vit_feature_difToken_AISO as models_vit_feature_difToken
from src import models_vit_feature_difToken_AISO_StracturePooling as models_vit_feature_difToken
from src.video_dataset_new import MaskMapGenerator

import mindspore.ops as ops
import numpy as np
from mindspore.common.tensor import Tensor
import cv2
from tools.ADtoolkit.video2numpy2video import numpy_array_to_video


def downloadmodel(use_ckpt, use_parallel, args):
    """
    :param use_ckpt: 等待提取的模型路径
    :param useParallel: 是否并行
    :return: 模型与运行环境参数
    """
    # parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    # 读入模型生成默认配置
    # parser.add_argument('--config_file', type=str,
    #                     default="config/jupyter_config/finetune/finetune_ViT-B-eval_copy.yaml")
    # args = parser.parse_args()
    # args = parser.parse_known_args()
    user_args = args
    args = get_config(args.config_file)

    # 以下修改实现了直接在get_run_args直接修改特征提取的所有参数，不再需要修改yaml文件
    if use_ckpt is not None:
        args.use_ckpt = use_ckpt
    if use_parallel is not None:
        args.use_parallel = use_parallel

    context_config = {
        "mode": args.mode,
        "device_target": args.device_target,
        "device_id": args.device_id,
        'max_call_depth': args.max_call_depth,
        'save_graphs': args.save_graphs,
    }
    parallel_config = {
        'parallel_mode': args.parallel_mode,
        'gradients_mean': args.gradients_mean,
    }
    local_rank, device_id, device_num = cloud_context_init(seed=args.seed,
                                                           use_parallel=args.use_parallel,
                                                           context_config=context_config,
                                                           parallel_config=parallel_config)
    print(f"local_rank: {local_rank}, device_num: {device_num}, device_id: {device_id}")
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)
    # create model
    args.logger.info("Create model...")
    # assert user_args.mask_ratio > 0
    print("mask_ratio: ", str(user_args.mask_ratio))
    args.mask_ratio = user_args.mask_ratio
    args.alpha = user_args.alpha
    # model = models_vit_feature.VisionTransformer_v2(**vars(args))
    model = models_vit_feature_difToken.VisionTransformer_v2(**vars(args))
    size = ops.Size()
    n_parameters = sum(size(p) for p in model.trainable_params() if p.requires_grad)
    args.logger.info("number of params: {}".format(n_parameters))

    # load pretrain ckpt
    if args.use_ckpt:
        args.logger.info(f"Load ckpt: {args.use_ckpt}...")
        params_dict = load_checkpoint(args.use_ckpt)
        msg = load_param_into_net(model, params_dict)
        if len(msg):
            args.logger.info(msg)
        else:
            args.logger.info("All keys match successfully!")
    model.set_train(False)
    return model, local_rank, device_id, device_num


class DataLoader:
    def __init__(self, resize_op_aim=14):
        self.resize = vision.Resize(resize_op_aim, Inter.BICUBIC)  # resoze to (14,14,C) or (14,14)

    @staticmethod
    def load_frame(frame_file):
        data = Image.open(frame_file)
        data = data.resize((340, 256), Image.ANTIALIAS)

        data = np.array(data)
        data = data.astype(float)
        data = (data * 2 / 255) - 1

        assert (data.max() <= 1.0) and (data.min() >= -1.0)

        return data  # return a frame(np.array)

    @staticmethod
    def load_rgb_batch(frames_dir, rgb_files, frame_indices):
        batch_data = np.zeros(frame_indices.shape + (256, 340, 3))
        for i in range(frame_indices.shape[0]):
            for j in range(frame_indices.shape[1]):
                batch_data[i, j, :, :, :] = DataLoader.load_frame(
                    os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))

        return batch_data  # shape:(batch_size,num_frames,256, 340, 3) 我猜的

    @staticmethod
    def oversample_data(data: np.ndarray):  # (39, 16, 224, 224, 2)  # 10 crop
        ###---------------I3D model to extract snippet feature---------------------
        # Input:  bx3x16x224x224  np.ndarray
        # Output: bx1024  np.ndarray
        data_flip = data[:, :, :, ::-1, :]

        data_1 = data[:, :, :224, :224, :]
        data_2 = data[:, :, :224, -224:, :]
        data_3 = data[:, :, 16:240, 58:282, :]
        data_4 = data[:, :, -224:, :224, :]
        data_5 = data[:, :, -224:, -224:, :]

        data_f_1 = data_flip[:, :, :224, :224, :]
        data_f_2 = data_flip[:, :, :224, -224:, :]
        data_f_3 = data_flip[:, :, 16:240, 58:282, :]
        data_f_4 = data_flip[:, :, -224:, :224, :]
        data_f_5 = data_flip[:, :, -224:, -224:, :]

        return [
            data_1, data_2, data_3, data_4, data_5,
            data_f_1, data_f_2, data_f_3, data_f_4, data_f_5
        ]

    @staticmethod
    def oversample_data_1crop(video_data, target_size=(224, 224, 3)):  # (39, 16, 224, 224, 3)
        assert target_size[0] == target_size[1]  # 目前之设计了针对正方形裁切
        # 获取输入数据的维度
        B, T, h, w, c = video_data.shape

        # 计算裁切的起始位置
        crop_size = min(h, w)
        if h > w:
            start_h = (h - crop_size) // 2
            start_w = 0
        else:
            start_h = 0
            start_w = (w - crop_size) // 2

        # 裁切视频数据
        cropped_data = video_data[:, :, start_h:start_h + crop_size, start_w:start_w + crop_size,
                       :]  # (B, T, 224, 224, 3)

        # 创建用于存储调整大小后的视频数据的数组
        resized_data = np.zeros((B * T, target_size[0], target_size[1], target_size[2]))
        cropped_data = np.resize(cropped_data, (B * T, crop_size, crop_size, 3))

        # resize每一帧
        for t in range(cropped_data.shape[0]):  # B * T
            frame = cropped_data[t]  # 获取当前帧
            resized_frame = cv2.resize(frame, (target_size[0], target_size[1]))  # 调整当前帧大小 ->(224,224,3)
            resized_data[t] = resized_frame  # 存储调整大小后的帧

        resized_data = np.resize(resized_data, (B, T, target_size[0], target_size[1], target_size[2]))
        # resized_data = np.resize(cropped_data, (B, T, crop_size, crop_size, target_size[2]))
        return [resized_data]


class DataLoader_TAD:

    def __init__(self, resize_op_aim=14):
        self.resize = vision.Resize(resize_op_aim, Inter.BICUBIC)  # resoze to (14,14,C) or (14,14)

    def test(self):
        test_video_folder = '/home/ma-user/work/dataset/TAD/frames/01_Accident_001.mp4/'
        test_frame = '1.jpg'
        frame_path = test_video_folder + test_frame
        output_path = '/home/ma-user/work/dataset/TAD/videoResizeTest/'
        data = self.load_frame(frame_path)

        # process Img
        data = Image.open(test_video_folder)
        data.save(output_path, test_frame)  # raw frame

        data = data.resize((340, 256), Image.ANTIALIAS)
        # data = np.array(data)
        data.save(output_path, test_frame.replace('.jpg', '') + '_resized.jpg')  # raw frame

        # Save the processed image
        # processed_image = Image.fromarray(data)
        # processed_image.save('path_to_save_image/processed_image.jpg')

    @staticmethod
    def load_frame(frame_file):
        data = Image.open(frame_file)
        data = data.resize((340, 256), Image.ANTIALIAS)

        data = np.array(data)
        data = data.astype(float)
        data = (data * 2 / 255) - 1

        assert (data.max() <= 1.0) and (data.min() >= -1.0)

        return data  # return a frame(np.array)

    @staticmethod
    def load_rgb_batch(frames_dir, rgb_files, frame_indices):
        batch_data = np.zeros(frame_indices.shape + (256, 340, 3))
        for i in range(frame_indices.shape[0]):
            for j in range(frame_indices.shape[1]):
                batch_data[i, j, :, :, :] = DataLoader.load_frame(
                    os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))

        return batch_data  # shape:(batch_size,num_frames,256, 340, 3) 我猜的

    @staticmethod
    def oversample_data(data: np.ndarray):  # (39, 16, 224, 224, 2)  # 10 crop
        ###---------------I3D model to extract snippet feature---------------------
        # Input:  bx3x16x224x224  np.ndarray
        # Output: bx1024  np.ndarray
        data_flip = data[:, :, :, ::-1, :]

        data_1 = data[:, :, :224, :224, :]
        data_2 = data[:, :, :224, -224:, :]
        data_3 = data[:, :, 16:240, 58:282, :]
        data_4 = data[:, :, -224:, :224, :]
        data_5 = data[:, :, -224:, -224:, :]

        data_f_1 = data_flip[:, :, :224, :224, :]
        data_f_2 = data_flip[:, :, :224, -224:, :]
        data_f_3 = data_flip[:, :, 16:240, 58:282, :]
        data_f_4 = data_flip[:, :, -224:, :224, :]
        data_f_5 = data_flip[:, :, -224:, -224:, :]

        return [
            data_1, data_2, data_3, data_4, data_5,
            data_f_1, data_f_2, data_f_3, data_f_4, data_f_5
        ]

    @staticmethod
    def oversample_data_1crop(video_data, target_size=(224, 224, 3)):  # (39, 16, 224, 224, 3)
        assert target_size[0] == target_size[1]  # 目前之设计了针对正方形裁切
        # 获取输入数据的维度
        B, T, h, w, c = video_data.shape

        # 计算裁切的起始位置
        crop_size = min(h, w)
        if h > w:
            start_h = (h - crop_size) // 2
            start_w = 0
        else:
            start_h = 0
            start_w = (w - crop_size) // 2

        # 裁切视频数据
        cropped_data = video_data[:, :, start_h:start_h + crop_size, start_w:start_w + crop_size,
                       :]  # (B, T, 224, 224, 3)

        # 创建用于存储调整大小后的视频数据的数组
        resized_data = np.zeros((B * T, target_size[0], target_size[1], target_size[2]))
        cropped_data = np.resize(cropped_data, (B * T, crop_size, crop_size, 3))

        # resize每一帧
        for t in range(cropped_data.shape[0]):  # B * T
            frame = cropped_data[t]  # 获取当前帧
            resized_frame = cv2.resize(frame, (target_size[0], target_size[1]))  # 调整当前帧大小 ->(224,224,3)
            resized_data[t] = resized_frame  # 存储调整大小后的帧

        resized_data = np.resize(resized_data, (B, T, target_size[0], target_size[1], target_size[2]))
        # resized_data = np.resize(cropped_data, (B, T, crop_size, crop_size, target_size[2]))
        return [resized_data]


class Task_collector:
    """
    用于获取待处理视频列表
    """

    @staticmethod
    def assign_tasks(tasks, processors=8):
        """
        :param processors: int num of processors
        :param tasks: task_list ,such as: [("任务1", 5), ("任务2", 8), ("任务3", 3), ("任务4", 2), ("任务5", 6)]
        :return: processor_tasks, load_difference ,max_load
                 各处理器任务列表 ，  最大任务量差异 ， 最大任务量
        """
        sorted_tasks = sorted(tasks, key=lambda x: x[1], reverse=True)  # reverse=True 推理， False调试
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

    @staticmethod
    def count_files_in_folder(folder_path):  # 输入一个文件夹路径，返回其中的文件数量
        count = 0
        for _, _, files in os.walk(folder_path):
            count += len(files)
        return count

    @staticmethod
    def get_video_list_with_framenum_all(input_dir):  # input_dir:'/home/ma-user/work/dataset/ucf_crime/frames'
        video_list = []  # 包含所有视频文件的列表
        cost_list = []
        for video in os.listdir(input_dir):
            video_path = os.path.join(input_dir, video)
            cost_list.append(Task_collector.count_files_in_folder(video_path))
            video_list.append(video_path)
        print("get {} videos".format(len(video_list)))
        return zip(video_list, cost_list)

    @staticmethod
    def get_video_list_with_framenum_continue(input_dir, output_dir):  # 搜集未处理的任务
        video_list = []  # 包含所有视频文件的列表
        cost_list = []
        # passlist = []  # extract_all_video
        # passlist=['Normal_Videos307_x264','Normal_Videos308_x264','Normal_Videos549_x264','Normal_Videos633_x264'] #ucf
        passlist=['Normal_Videos307_x264','Normal_Videos308_x264','Normal_Videos633_x264','.ipynb_checkpoints'] #ucf

        for video in os.listdir(input_dir):
            if video in passlist:
                continue
            save_file = '{}_{}.npy'.format(video, "videomae")
            if save_file in os.listdir(os.path.join(output_dir)):
                # print("{} has been extracted".format(save_file))
                pass
            else:
                video_path = os.path.join(input_dir, video)
                cost_list.append(Task_collector.count_files_in_folder(video_path))
                video_list.append(video_path)
        print("leave {} videos".format(len(video_list)))
        return zip(video_list, cost_list)  # 任务列表与预算


def forward_batch(b_data, net, ids_keep=None, depth=None):  # b_data:b,c,t,h,w
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = Tensor(b_data)  # b,c,t,h,w  # bx3x16x224x224
    if ids_keep is not None:
        ids_keep = Tensor(ids_keep)  # B, L_keep, C
    if depth is not None:
        depth = Tensor(depth)
    b_features = net(b_data, ids_keep, depth)
    b_features = b_features.asnumpy()
    return b_features


def run(args_item):
    # get run config
    video_dir, run_config_dic = args_item
    output_dir = run_config_dic['output_dir']
    batch_size = run_config_dic['batch_size']
    crop_num = run_config_dic['crop_num']
    dataset = run_config_dic['dataset']
    mask_ratio = run_config_dic['mask_ratio']

    video_dir_depth = video_dir.replace("/frames", "/DepthFrames").replace(".mp4", "")  # frames 与 depthframes位于同一目录下
    dataloader = DataLoader()

    chunk_size = 16
    frequency = 16
    maskMapGenerator = MaskMapGenerator(patch_num=14, mask_ratio=mask_ratio)  # mask_ratio：0.5(best)/0.9/0.75
    video_name = video_dir.split("/")[-1]
    video_name = video_name.replace(".mp4", "")
    save_file = '{}_{}.npy'.format(video_name, "videomae")
    if save_file in os.listdir(os.path.join(output_dir)):
        print("{} has been extracted".format(save_file))
        return

    # set up the model
    rgb_files = [i for i in os.listdir(video_dir) if i.endswith('jpg')]
    rgb_files_depth = [i for i in os.listdir(video_dir_depth) if i.endswith('jpg')]
    if dataset == "UCF-Crime":
        rgb_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))  # ucf
        rgb_files_depth.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))  # ucf
    elif dataset == "TAD":
        rgb_files.sort(key=lambda x: int(x.split(".")[0]))  # TAD
        rgb_files_depth.sort(key=lambda x: int(x.split(".")[0]))  # TAD
    frame_cnt = len(rgb_files)
    frame_cnt_depth = len(rgb_files_depth)

    if not frame_cnt > chunk_size:
        print("{} has not enough frames".format(video_dir))
        return
    # if not frame_cnt == frame_cnt_depth:
    #     print("frame_cnt!=frame_cnt_depth".format(video_dir))
    #     return
    # assert (frame_cnt > chunk_size), print("{} has not enough frames".format(video_dir))
    # assert frame_cnt == frame_cnt_depth, print("frame_cnt!=frame_cnt_depth".format(video_dir))

    # the number of chunks that can be formed from the images in the folder.
    # If there are not enough images, copy the last frames to fill the gap.
    clipped_cnt = math.ceil(frame_cnt / chunk_size)
    copy_length = (clipped_cnt * frequency) - frame_cnt
    if copy_length != 0:
        copy_imgs = [rgb_files[frame_cnt - 1]] * copy_length
        rgb_files = rgb_files + copy_imgs
        copy_imgs_depth = [rgb_files_depth[frame_cnt_depth - 1]] * copy_length
        rgb_files_depth = rgb_files_depth + copy_imgs_depth

    frame_indices = []  # Frames to chunks,contain lists of indices of each chunk,such as:[[0,1,...15][16,...,31]...]
    for i in range(0, clipped_cnt):
        frame_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])
    frame_indices = np.array(frame_indices)
    # assert frame_indices.shape[0] == clipped_cnt

    batch_num = int(np.ceil(clipped_cnt / batch_size))
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)

    full_features = [[] for i in range(crop_num)]

    for batch_id in tqdm(range(batch_num)):
        # start = time.time()
        batch_data = DataLoader.load_rgb_batch(video_dir, rgb_files, frame_indices[batch_id])
        try:
            batch_data_depth = DataLoader.load_rgb_batch(video_dir_depth, rgb_files_depth, frame_indices[batch_id])
        except:
            print('error folder:', video_dir_depth)

        batch_data_n_crop = []
        batch_data_n_crop_depth = []
        if crop_num == 1:
            batch_data_n_crop = DataLoader.oversample_data_1crop(batch_data)
            batch_data_n_crop_depth = DataLoader.oversample_data_1crop(batch_data_depth)
        elif crop_num == 10:
            batch_data_n_crop = DataLoader.oversample_data(batch_data)  # batch_data_n_crop ：[crop,b,r,h,w,c]
            batch_data_n_crop_depth = DataLoader.oversample_data(batch_data_depth)
        batch_data_n_crop_unnormalized = []  # 存放未归一化的原图
        batch_data_n_crop_unnormalized_depth = []  # 存放未归一化的原图
        for normalized_data in batch_data_n_crop:
            batch_data_n_crop_unnormalized.append(((normalized_data + 1) * (255 / 2)).astype(np.float32))  # 逆归一化操作
        for normalized_data_depth in batch_data_n_crop_depth:
            batch_data_n_crop_unnormalized_depth.append(
                ((normalized_data_depth + 1) * (255 / 2)).astype(np.float32))  # 逆归一化操作
        # end = time.time()
        # tqdm.write(f'DataPrepare_timecost:{end - start}s')

        # start = time.time()
        for i in range(crop_num):  # b,t,h,w,c
            assert (batch_data_n_crop[i].shape[-2] == 224) and (batch_data_n_crop[i].shape[-3] == 224)
            assert (batch_data_n_crop_depth[i].shape[-2] == 224) and (batch_data_n_crop_depth[i].shape[-3] == 224)

            # 取得灰度图
            frames_gray = maskMapGenerator.rgb_to_gray_cv2(batch_data_n_crop_unnormalized[i])  # frames_gray: b, t, h, w
            frames_gray = np.squeeze(frames_gray)
            frames_gray_depth = maskMapGenerator.rgb_to_gray_cv2(
                batch_data_n_crop_unnormalized_depth[i])  # frames_gray_depth: b, t, h, w
            frames_gray_depth = np.squeeze(frames_gray_depth)
            # frames_gray_depth=dataloader.resize(frames_gray_depth)

            # down sample depth map to (B,T,14,14)
            B, T, H, W = frames_gray_depth.shape
            frames_gray_depth = frames_gray_depth.reshape(B * T, H, W).transpose((1, 2, 0))  # H,W,B*T
            frames_gray_depth = dataloader.resize(frames_gray_depth)
            frames_gray_depth = frames_gray_depth.transpose((2, 0, 1)).reshape(B, T, 14, 14)  # B*T,14,14
            frames_gray_depth = (255 - frames_gray_depth) * 0.4 / 255.0 + 0.8  # weight in [0.5,1.5] , 1.5 is far
            frames_gray_depth = frames_gray_depth[:, ::2, :, :]  # B,8,14,14

            mask_prob = []
            b, t, h, w = frames_gray.shape
            for j in range(b):
                # mask_prob.append(maskMapGenerator.mask_probability_single(frames_gray[j],random_crop=False))
                mask_prob.append(
                    maskMapGenerator.mask_probability_single_dinamic_token(frames_gray[j], random_crop=False))
            mask_prob = np.array(mask_prob)  # (b,t,14,14)

            ids_keep = []
            for j in range(b):
                _mask, _ids_restore, _ids_keep, _ids_mask = \
                    maskMapGenerator.mask_probability_generator(mask_prob_frame=mask_prob[j], mask_prob_depth=None)
                ids_keep.append(_ids_keep)
            ids_keep = np.array(ids_keep)  # ids_keep : B, L_keep, C

            # for i,mov in enumerate(batch_data_n_crop_unnormalized_depth):
            #     numpy_array_to_video(mov[0],f"{video_name}_depth_crop{i}.mp4")

            # infer_start = time.time()
            full_features[i].append(forward_batch(batch_data_n_crop[i], model, ids_keep, frames_gray_depth))
            # infer_end = time.time()
            # if i == 0: tqdm.write(f'Infer1Crop_timecost:{infer_end - infer_start}s')

        # end = time.time()
        # tqdm.write(f'Inference_timecost:{end - start}s')

    full_features = [np.concatenate(i, axis=0) for i in full_features]  # 合并batch
    full_features = [np.expand_dims(i, axis=0) for i in full_features]
    full_features = np.concatenate(full_features, axis=0)  # 合并crop
    np.save(os.path.join(output_dir, save_file), full_features)

    print('{} done: {} / {}, {}'.format(video_name, frame_cnt, clipped_cnt, full_features.shape))


def get_run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="rgb", type=str)
    parser.add_argument('--use_ckpt', default="/home/ma-user/work/ckpt/9-5_9-1_finetune.ckpt",
                        type=str)  # 9-5_9-1_finetune.ckpt/11-1_10-15_finetune_L.ckpt
    # 9-1_pretrain_frame_with_depth_add_loss_on_surveillance_20w_change_encoder.ckpt  (waiting)  (记得改输出路径)

    parser.add_argument('--mask_ratio', default=0.25, type=float)  # 0.5
    parser.add_argument('--alpha', default=0.1, type=float, help="alpha is structure weight scaling factor")  # 0.5
    # parser.add_argument('--dataset', default="UCF-Crime", choices=['UCF-Crime', 'TAD', 'ShanghaiTech'], type=str)
    parser.add_argument('--use_depth', default=True, type=bool, help="use structure pooling")

    # ShanghaiTech
    # parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/ShanghaiTech/frames", type=str)
    # parser.add_argument('--output_dir',default="/home/ma-user/work/features/SHT_9-5_9-1_finetune_AISO_0.5",type=str)
    # UCF-Crime
    parser.add_argument('--dataset', default="UCF-Crime", choices=['UCF-Crime', 'TAD', 'ShanghaiTech'], type=str)
    # parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/ucf_crime/frames", type=str)
    parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/ucf_crime_less4/frames", type=str)
    parser.add_argument('--output_dir',
                        default="/home/ma-user/work/features/UCF_9-5_9-1_finetune_DT0.25_SPa0.1/less3",  # test
                        type=str)

    # parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/ucf_crime_less4/frames", type=str)
    # parser.add_argument('--output_dir',
    #                     default="/home/ma-user/work/features/UCF_9-1_finetune",  # test
    #                     type=str)
    # parser.add_argument('--output_dir',
    #                     default="/home/ma-user/work/features/test",  # test
    #                     type=str)
    # parser.add_argument('--node_id',default=0,type=int)

    # XD-Violence
    # parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/XD-Violence/test/frames", type=str)
    # parser.add_argument('--output_dir',
    #                     default="/home/ma-user/work/features/featureTest/test",
    #                     type=str)
    # TAD
    # parser.add_argument('--dataset', default="TAD", choices=['UCF-Crime', 'TAD', 'ShanghaiTech'], type=str)
    # parser.add_argument('--input_dir', default="/home/ma-user/work/dataset/TAD/frames", type=str)
    # parser.add_argument('--output_dir', default="/home/ma-user/work/features/test",
    #                     type=str)  # test

    parser.add_argument('--use_parallel', default=True, type=bool)  # True / Falses
    parser.add_argument('--crop_num', type=int, default=10, choices=[1, 10])  # 1/10
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_mode', default="oversample", type=str)
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--config_file', type=str,
                        default="config/jupyter_config/finetune/finetune_ViT-B-eval_copy.yaml")  # finetune_ViT-L-eval.yaml / finetune_ViT-B-eval_copy.yaml

    run_args = parser.parse_args()

    return run_args


# def get_node_id():
#     root="/home/ma-user/work/B"
#     for i in range(8):
#         if os.path.exists(root + str(i)):
#             return i
#     else:
#         raise RuntimeError

if __name__ == '__main__':
    # node_id=get_node_id()

    run_args = get_run_args()

    model, local_rank, device_id, device_num = downloadmodel(use_ckpt=run_args.use_ckpt,
                                                             use_parallel=run_args.use_parallel, args=run_args)
    print(f"local_rank:{local_rank}, device_id:{device_id},device_num:{device_num}")
    os.makedirs(run_args.output_dir, exist_ok=True)
    # device_id+=8*(node_id)

    task_list = Task_collector.get_video_list_with_framenum_continue(run_args.input_dir, run_args.output_dir)
    # processor_tasks, load_difference, max_load = Task_collector.assign_tasks(task_list,processors=device_num)  # result: 2d task list
    processor_tasks, load_difference, max_load = Task_collector.assign_tasks(task_list,
                                                                             processors=8)  # result: 2d task list

    print(f"max load:{max_load},max load diff：{load_difference}")

    # 使用嵌套的列表推导式提取每个元组的第一个元素
    task_list = [[item[0] for item in sublist] for sublist in processor_tasks]
    local_task_list = task_list[device_id]

    nums = len(local_task_list)
    run_config_dic = {'output_dir': run_args.output_dir, 'batch_size': run_args.batch_size,
                      'crop_num': run_args.crop_num,
                      'dataset': run_args.dataset, 'mask_ratio': run_args.mask_ratio, 'use_depth': run_args.use_depth}
    for inputs in zip(local_task_list, [run_config_dic] * nums):
        run(inputs)
