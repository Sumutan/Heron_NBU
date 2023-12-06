import multiprocessing
import os

import cv2
import numpy as np
import random

from mindspore import dataset
from mindspore.dataset import (DistributedSampler, SequentialSampler,
                               transforms, vision)
from mindspore.dataset.vision import Inter

from transforms.mixup import Mixup
from utils_mae import video_utils


class VideoDataset:
    """
    提供了构建用于K400视频数据集的公共类
    添加了采集帧差图与深度图的实现
    """

    def __init__(self,
                 mode,
                 path_to_data_dir,
                 logger,

                 # decoding setting
                 sampling_rate=4,
                 num_frames=16,
                 input_size=224,

                 # pretrain augmentation
                 repeat_aug=1,

                 # other parameters
                 use_offset_sampling=True,
                 num_retries=10,
                 ):
        self.mode = mode
        self._path_to_data_dir = path_to_data_dir

        self.logger = logger

        self._repeat_aug = repeat_aug
        self._num_retries = num_retries

        self._input_size = input_size

        self._sampling_rate = sampling_rate
        self._num_frames = num_frames
        self._clip_size = self._sampling_rate * self._num_frames

        self._use_offset_sampling = use_offset_sampling
        self.logger.info("Constructing Videos {}...".format(mode))

        self.init_check()

    def init_check(self):
        assert self.mode in ['pretrain', 'finetune', 'val', 'test'], f"{self.mode}"
        assert os.path.isdir(self._path_to_data_dir), f"{self._path_to_data_dir}"

    def _construct_loader(self):
        """
        Construct the video loader.
        构建视频加载器。
        """
        # 定义一个字典，存储不同模式下的CSV文件名
        csv_file_name = {
            "pretrain": "train",
            "finetune": "train",
            "val": "val",
            "test": "test",
        }
        # 拼接CSV文件的路径
        path_to_file = os.path.join(
            self._path_to_data_dir,
            "{}.csv".format(csv_file_name[self.mode]),
        )
        # 断言CSV文件是否存在
        assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)

        # 初始化存储视频路径、标签和空间-时间索引的列表
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 2, path_label
                path, label = path_label.split()
                for idx in range(self._num_clips):
                    self._path_to_videos.append(os.path.join(path))
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)

        # 断言视频路径列表不为空
        assert len(self._path_to_videos) > 0, "Failed to load Kinetics from {}".format(path_to_file)

        # 输出日志信息，显示构建的视频数据加载器的大小和路径
        self.logger.info("Constructing video dataloader (size: {}) from {}".format(
            len(self._path_to_videos), path_to_file)
        )

    def _construct_loader_depth(self):
        # 定义一个字典，存储不同模式下的CSV文件名
        csv_file_name = {
            "pretrain": "train",
            "finetune": "train",
            "val": "val",
            "test": "test",
        }
        path_to_file_depth = os.path.join(
            self._path_to_data_dir,  # /home/dxm/dataset/dataset/train_depth.scv
            "{}_depth.csv".format(csv_file_name[self.mode]),
        )

        assert os.path.exists(path_to_file_depth), "{} dir not found".format(path_to_file_depth)

        self._path_to_videos_depth = []
        self._labels_depth = []
        self._spatial_temporal_idx_depth = []

        with open(path_to_file_depth, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 2, path_label
                path, label = path_label.split()
                for idx in range(self._num_clips):
                    self._path_to_videos_depth.append(os.path.join(path))
                    self._labels_depth.append(int(label))
                    self._spatial_temporal_idx_depth.append(idx)

        assert len(self._path_to_videos_depth) > 0, "Failed to load Kinetics from {}".format(path_to_file_depth)

        self.logger.info("Constructing video dataloader (size: {}) from {}".format(
            len(self._path_to_videos_depth), path_to_file_depth)
        )

    def _get_video_cap(self, index):
        """
        :param index: 视频文件列表的下标索引
        :return:  cv2 cap，文件路径
        """
        for retry in range(self._num_retries):  # 视频打开失败则重新随机抓取视频
            # 获取视频路径
            sample = self._path_to_videos[index]
            if not os.path.exists(sample):
                self.logger.warning("文件{}不存在".format(sample))
                index = np.random.randint(0, len(self._path_to_videos) - 1)
                continue
            # 打开视频文件
            cap = cv2.VideoCapture(sample)
            if not cap.isOpened():
                self.logger.warning("视频打开失败,video id: {}, retries: {}, path: {}"
                                    .format(index, retry, sample))
                cap.release()
                index = np.random.randint(0, len(self._path_to_videos) - 1)
                continue

            return cap, sample, index  # 返回打开的视频对象和样本路径
        raise RuntimeError("数据多次读取失败！")  # 如果无法成功打开和读取视频，则触发运行时错误

    def _get_snippiets(self, cap, index):
        """
        用于在一段视频中采样一小段
        这个函数返回类别标签，所以应该被用在finetune阶段
        :param cap:  cv视频流
        :param index:  视频文件列表的下标索引
        :return:
            frame_list:[r,t,h,w,c]   一段视频切片，r为从一个视频中的重复采样次数
            label list: int          表示标签类别的int数
            success_sampled: True    执行成功标记
        """
        success_sampled = False
        total_frames = cap.get(7)  # 视频总帧数

        if self.mode in ["pretrain", "finetune"]:
            temporal_sample_index = -1  # -1 indicates random sampling.
        elif self.mode == "val":
            temporal_sample_index = 0
        elif self.mode == "test":
            temporal_sample_index = self._spatial_temporal_idx[index] // self._test_num_spatial_crops
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        frame_list = []
        label_list = []
        label = self._labels[index]
        for i in range(self._repeat_aug):
            # try selective decoding.
            video_start_frame, video_end_frame = video_utils.get_start_end_idx(
                total_frames,
                self._clip_size,
                temporal_sample_index,
                self._test_num_ensemble_views,
                use_offset=self._use_offset_sampling,
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

            frames = []
            # 读取指定数量的帧
            for j in range(self._clip_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            frames = np.array(frames)
            try:
                frames = video_utils.temporal_sampling(frames, 0, self._clip_size - 1, self._num_frames)
                frame_list.append(frames)
                label_list.append(label)
                if len(frame_list) == self._repeat_aug:
                    success_sampled = True
            except:
                sample = self._path_to_videos[index]
                self.logger.warning(f"temporal sampling failed,file:{sample}")
                cap.release()
                index = np.random.randint(0, len(self._path_to_videos) - 1)
                sample = self._path_to_videos[index]
                cap = cv2.VideoCapture(sample)

        frame_list = np.array(frame_list)
        label_list = np.array(label_list, dtype=np.int32)

        cap.release()

        return frame_list, label_list, success_sampled

    def _get_snippietsAndframeDif(self, cap, index):
        """
        用于在一段视频中采样一小段
        *似乎这段代码设计的时候只考虑r=1的情况，启用时要禁用视频重采样
        input:
            :param index:  视频文件列表的下标索引
            :param start_idx: 开始帧的id
            :param end_idx: 结束帧的id，start_idx与end_idx在之前调用_get_snippiets时得到
            :param frames_gray_list: 上一个环节得到的视频帧的灰度图列表
        :return:
            frames_list frame_list:[r,t,h,w,c]   一段视频切片，r为从一个视频中的重复采样次数
            frames_depth_gray_list: [t,h,w]   深度图列表
            start_idx, end_idx: 后续提取深度可能用到的位置标记
            success_sampled: True    执行成功标记
        """
        success_sampled = False
        total_frames = cap.get(7)  # 视频总帧数

        if self.mode in ["pretrain", "finetune"]:
            temporal_sample_index = -1  # -1 indicates random sampling.
        elif self.mode == "val":
            temporal_sample_index = 0
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        frames_list = []
        frames_gray_list = []
        start_idx, end_idx = 0, 0
        for i in range(self._repeat_aug):
            # try selective decoding.
            start_idx, end_idx = video_utils.get_start_end_idx(
                total_frames,
                self._clip_size,
                temporal_sample_index,
                0,
                use_offset=self._use_offset_sampling,
            )

            video_start_frame = int(start_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

            frames, frames_gray = [], []
            # 读取指定数量的帧
            for j in range(self._clip_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame_rgb)
                frames_gray.append(frame_gray)

            frames = np.array(frames)
            frames_gray = np.array(frames_gray)

            try:
                frames = video_utils.temporal_sampling(frames, 0, self._clip_size - 1,
                                                       self._num_frames)
                frames_list.append(frames)
                frames_gray = video_utils.temporal_sampling_gray(frames_gray, 0, self._clip_size - 1,
                                                                 self._num_frames)
                frames_gray_list.append(frames_gray)
                if len(frames_list) == self._repeat_aug:
                    success_sampled = True
                    cap.release()
            except:
                sample = self._path_to_videos[index]
                self.logger.warning(f"temporal sampling failed,file:{sample}")
                success_sampled = False

        return frames_list, frames_gray_list, start_idx, end_idx, success_sampled

    def _get_snippiets_depth(self, index, start_idx, end_idx, frames_gray_list):
        """
        用于在一段视频中采样一小段
        *似乎这段代码设计的时候只考虑r=1的情况，启用时要禁用视频重采样
        input:
            :param index:  视频文件列表的下标索引
            :param start_idx: 开始帧的id
            :param end_idx: 结束帧的id，start_idx与end_idx在之前调用_get_snippiets时得到
            :param frames_gray_list: 上一个环节得到的视频帧的灰度图列表
        :return:
            frames_depth_gray_list: [t,h,w]   深度图列表
            success_sampled: True    执行成功标记
        """
        # depth
        # 对应原始视频64帧的  7帧深度图的起始id
        start_idx_d = int(start_idx / 10)  # 深度图提取的采样率是1/10，这里对齐下标
        end_idx_d = int(end_idx / 10)
        clip_size_d = int(end_idx_d - start_idx_d) + 1

        sample_depth = self._path_to_videos_depth[index]
        cap_d = cv2.VideoCapture(sample_depth)
        success_sampled = False

        frames_depth_gray_list = []
        for i in range(self._repeat_aug):
            # try selective decoding.
            video_start_frame = int(start_idx_d)
            # 设置读取帧的位置
            cap_d.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

            frames_depth_gray = []

            # 读取指定数量的帧  7
            for j in range(clip_size_d):
                ret, frame = cap_d.read()
                if not ret:
                    break
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames_depth_gray.append(frame_gray)
            frames_depth_gray = np.array(frames_depth_gray)  # (7, 250, 490)

            # 7帧 扩展 16 帧
            try:
                for j in range(self._num_frames):
                    start_id = int((start_idx + int(4 * j)) / 10)
                    frames_depth_gray_list.append(frames_depth_gray[start_id - start_idx_d])
                if len(frames_gray_list) == self._repeat_aug:
                    success_sampled = True
            except:
                self.logger.warning(f"temporal sampling failed,file:{sample_depth}")
                raise RuntimeError

            cap_d.release()

        return frames_depth_gray_list, success_sampled


class MaskMapGenerator:
    """
    提供了深度图与帧差提示的掩码策略的一些方法
    """

    def __init__(self, testMode=False, patch_num=None, mask_ratio=None):
        if patch_num:
            self._patch_num = patch_num
        if mask_ratio is not None:
            self._mask_ratio = mask_ratio
        if testMode:
            return

        # assert self._patch_num > 0, "path_num must be greater than 0"

    def get_resize_op(self, random_crop=True):
        if random_crop:
            # 原始宽高的帧差resize到（224, 224)
            # ori -->(16, 250, 490) --->(16, 224, 224)
            resize_op = vision.RandomResizedCrop(size=224, scale=(0.5, 1.0), ratio=(0.75, 1.333))
        else:
            resize_op = vision.Resize(size=224)
        return resize_op

    def rgb_to_gray_np(self, rgb_images):
        # 计算灰度图像素值
        gray_images = np.dot(rgb_images[..., :3], [0.2989, 0.5870, 0.1140])

        # 转换数据类型为整数
        gray_images = gray_images.astype(np.uint8)

        return gray_images

    def rgb_to_gray_cv2(self, rgb_images):
        # 获取输入图片序列的形状信息
        b, t, h, w, c = rgb_images.shape

        # 创建用于存储灰度图的数组
        gray_images = np.zeros((b, t, h, w, 1))

        # 遍历每个图片，进行灰度化转换
        for i in range(b):
            for j in range(t):
                # 使用OpenCV将彩色图转换为灰度图
                gray_images[i, j, :, :, 0] = cv2.cvtColor(rgb_images[i, j, :, :, :], cv2.COLOR_RGB2GRAY)

        return gray_images

    def mask_probability_single_dinamic_token(self, frames_gray, mask_map_size=(14, 14), random_crop=False):
        """
        单帧差法生成mask 概率图:
        有限输出所有高概率部分
        """
        # shape transform -> frames_gray: (16, 320, 426)
        if len(frames_gray.shape) == 4:
            r, t, h, w, = frames_gray.shape  # (1, 16, 320, 426)
            assert r == 1  # 目前代码只考虑到r==1的情况
        elif len(frames_gray.shape) == 3:
            t, h, w, = frames_gray.shape  # ( 16, 320, 426)
        else:
            raise ValueError("frames_gray shape error")
        frames_gray = frames_gray.reshape(t, h, w)  # (16, 320, 426)

        # 帧差计算
        gray_lwpCV = frames_gray[0]
        gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)  # 对转换后的灰度图进行高斯模糊
        background = gray_lwpCV  # 将高斯模糊后的第一帧作为初始化背景

        diff_list = []
        index = np.array(range(1, t))
        # print(index)
        for i in index:
            gray_CV = frames_gray[i]
            gray_CV = cv2.GaussianBlur(gray_CV, (21, 21), 0)
            diff = cv2.absdiff(background, gray_CV)  # 将最新读取的视频帧和背景做差
            diff_list.append(diff)

        diff_list.append(diff)
        # 存储了16帧的帧差信息，15,16帧为同一个
        diff_list_0 = np.array(diff_list)

        resize_op = self.get_resize_op(random_crop)

        diff_list_done = []
        for i in range(t):
            diff_list1 = resize_op(diff_list_0[i])
            diff_list_done.append(diff_list1)
        diff_list_done = np.array(diff_list_done)  # shape:(16, 224, 224)
        assert diff_list_done.shape == (t, 224, 224)


        # patch:16*16 帧差下采样 (16, 224, 224)--->(16, 14, 14)
        patch_prob = []
        threshold = int(10)
        for diff in diff_list_done:
            patch = cv2.resize(diff, mask_map_size, interpolation=cv2.INTER_LINEAR)  # 下采样

            patch = np.where(patch == 0, 1e-5, patch)
            patch_prob.append(patch)

        patch_prob = np.array(patch_prob)  # shape:(16, 14, 14)

        # 生成概率图 归一化
        mask_prob = []
        for patch in patch_prob:
            prob = patch / np.sum(patch)
            mask_prob.append(prob)

        mask_prob = np.array(mask_prob)  # (16, 14, 14)

        return mask_prob

    def mask_probability_single(self, frames_gray, mask_map_size=(14, 14), random_crop=True):
        """
        单帧差法生成mask 概率图:
        1.做侦察图后进行降采样，取得（t,14,14）的帧差缩略图
        2.
        """
        # shape transform -> frames_gray: (16, 320, 426)
        if len(frames_gray.shape) == 4:
            r, t, h, w, = frames_gray.shape  # (1, 16, 320, 426)
            assert r == 1  # 目前代码只考虑到r==1的情况
        elif len(frames_gray.shape) == 3:
            t, h, w, = frames_gray.shape  # ( 16, 320, 426)
        else:
            raise ValueError("frames_gray shape error")
        frames_gray = frames_gray.reshape(t, h, w)  # (16, 320, 426)

        # 帧差计算
        gray_lwpCV = frames_gray[0]
        gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)  # 对转换后的灰度图进行高斯模糊
        background = gray_lwpCV  # 将高斯模糊后的第一帧作为初始化背景

        diff_list = []
        index = np.array(range(1, t))
        # print(index)
        for i in index:
            gray_CV = frames_gray[i]
            gray_CV = cv2.GaussianBlur(gray_CV, (21, 21), 0)
            diff = cv2.absdiff(background, gray_CV)  # 将最新读取的视频帧和背景做差
            diff_list.append(diff)

        diff_list.append(diff)
        # 存储了16帧的帧差信息，15,16帧为同一个
        diff_list_0 = np.array(diff_list)

        resize_op = self.get_resize_op(random_crop)

        diff_list_done = []
        for i in range(t):
            diff_list1 = resize_op(diff_list_0[i])
            diff_list_done.append(diff_list1)
        diff_list_done = np.array(diff_list_done)  # shape:(16, 224, 224)
        assert diff_list_done.shape == (t, 224, 224)

        # patch:16*16 帧差下采样 (16, 224, 224)--->(16, 14, 14)
        patch_prob = []
        threshold = int(10)
        for diff in diff_list_done:
            patch = cv2.resize(diff, mask_map_size, interpolation=cv2.INTER_LINEAR)  # 下采样

            # patch 为0值 随机设置（1，5）之间的值
            patch = np.where(patch == 0, np.random.randint(1, 5), patch)
            # 高概率区域保留25%， 将其概率值设置为0.1低概率值
            part1 = [(i, j) for i in range(len(patch)) for j in range(len(patch[i])) if
                     patch[i][j] >= threshold]  # 大于阈值（高概率、运动对象）

            num_selected = int(len(part1) * 0.25)
            select_indices = random.sample(part1, num_selected)
            for i, j in select_indices:
                patch[i][j] = 0
            patch = np.where(patch == 0, 1e-5, patch)
            patch_prob.append(patch)

        patch_prob = np.array(patch_prob)  # shape:(16, 14, 14)

        # 生成概率图 归一化
        mask_prob = []
        for patch in patch_prob:
            prob = patch / np.sum(patch)
            mask_prob.append(prob)

        mask_prob = np.array(mask_prob)  # (16, 14, 14)

        return mask_prob

    def mask_probability_depth(self, frames_depth_gray, mask_map_size=(14, 14), random_crop=False):
        r, t, h, w, = frames_depth_gray.shape  # 1, 16, 320, 426)
        frames_depth_gray = frames_depth_gray.reshape(t, h, w)  # (16, 320, 426)

        resize_op = self.get_resize_op(random_crop)

        depth_resize = []
        for i in range(t):
            resize = resize_op(frames_depth_gray[i])
            depth_resize.append(resize)

        depth_resize = np.array(depth_resize)  # (16, 224, 224)
        assert depth_resize.shape == (t, 224, 224)

        # patch:16*16 帧差下采样 (25, 224, 224)--->(25, 14, 14)
        depth_patch_prob = []
        # threshold = int(10)
        for frame in depth_resize:
            patch = cv2.resize(frame, mask_map_size, interpolation=cv2.INTER_LINEAR)  # 下采样
            # patch 为0值 随机设置（1，5）之间的值
            patch = np.where(patch == 0, np.random.randint(1, 5), patch)
            depth_patch_prob.append(patch)

        depth_patch_prob = np.array(depth_patch_prob)  # (16, 14, 14)

        # 生成概率图
        mask_prob_depth = []
        for patch in depth_patch_prob:
            prob = patch / np.sum(patch)
            mask_prob_depth.append(prob)

        mask_prob_depth = np.array(mask_prob_depth)  # (16, 14, 14)

        return mask_prob_depth

    def mask_probability_generator(self, mask_prob_frame, mask_prob_depth, mix_method="mul"):
        assert mask_prob_frame is not None or mask_prob_depth is not None
        if mix_method == "mul":
            if mask_prob_frame is None:
                mask_prob_frame = np.ones_like(mask_prob_depth)
            if mask_prob_depth is None:
                mask_prob_depth = np.ones_like(mask_prob_frame)
            prob_mul = mask_prob_frame * mask_prob_depth
        elif mix_method == "add":
            if mask_prob_frame is None:
                mask_prob_frame = np.zeros_like(mask_prob_depth)
            if mask_prob_depth is None:
                mask_prob_depth = np.zeros_like(mask_prob_frame)
            prob_mul = 0.7*mask_prob_frame + 0.3*mask_prob_depth
        else:
            raise ValueError

        mask_prob = []

        for p in prob_mul:
            prob = p / np.sum(p)
            mask_prob.append(prob)

        mask_prob = np.array(mask_prob)

        patch_num = int(self._patch_num ** 2)  # 196

        # 0：保留   1：mask
        # num_elements = int(self._mask_ratio * patch_num)
        num_elements = int(self._mask_ratio * patch_num)  # 75%: mask-176  save-20
        mask, ids_mask, ids_keep, ids_restore = [[] for i in range(4)]

        # mask_prob:(16, 14, 14)
        # 两帧采一帧
        j = 0
        for i in range(8):
            num_base = i * patch_num  # 第i帧的起始位置

            prob = mask_prob[j]
            prob_choice = prob.flatten()

            # 选取元素并将选取的元素置0
            # 可视化时 使用 np.ones       正常情况下使用 np.zeros
            mask_frame = np.zeros(shape=(14, 14))
            index = np.arange(patch_num)
            ids_mask_t = np.random.choice(index, size=num_elements, replace=False, p=prob_choice)
            # 可视化时  赋值为0，   正常情况下赋值为  1
            mask_frame.ravel()[ids_mask_t] = 1
            ids_keep_t = np.setdiff1d(index, ids_mask_t)
            mask_frame = mask_frame.reshape((14, 14))

            mask.append(mask_frame)
            ids_mask.append(ids_mask_t + num_base)
            ids_keep.append(ids_keep_t + num_base)
            ids_restore.append(np.concatenate(((ids_mask_t + num_base), (ids_keep_t + num_base)), axis=0))
            j += 2

        mask = np.array(mask).flatten()
        ids_keep = np.array(ids_keep).flatten()
        ids_mask = np.array(ids_mask).flatten()
        ids_restore = np.array(ids_restore).flatten()

        return mask, ids_restore, ids_keep, ids_mask


class VideoMaeDataset(VideoDataset, MaskMapGenerator):
    def __init__(
            self,
            mode,
            path_to_data_dir,
            logger,

            # decoding setting
            sampling_rate=4,
            num_frames=16,
            input_size=224,

            # other parameters
            use_offset_sampling=True,
            num_retries=10,

            # pretrain augmentation
            repeat_aug=1,

            # VideoMaeDataset set
            # random mask
            mask_strategy='random',  # random / DFDH / FDH
            t_patch_size=2,
            patch_size=16,  # the size of patch is 16*16*2
            mask_ratio=0.9,
    ):
        VideoDataset.__init__(
            self,
            mode=mode,
            path_to_data_dir=path_to_data_dir,
            logger=logger,
            sampling_rate=sampling_rate,
            num_frames=num_frames,
            repeat_aug=repeat_aug,
            num_retries=num_retries,
            input_size=input_size,
            use_offset_sampling=use_offset_sampling,
        )
        # For training or validation mode, one single clip is sampled from every video.
        # For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every video.
        # For every clip, NUM_SPATIAL_CROPS is cropped spatially from the frames.
        self._num_clips = 1

        self.logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        if mask_strategy == 'DFDH':
            self._construct_loader_depth()
            self.use_depth = True
        else:
            self.use_depth = False
        self._t_patch_size = t_patch_size
        self._patch_size = patch_size
        self._mask_ratio = mask_ratio
        self._patch_num = int(self._input_size // self._patch_size)

        assert self._num_frames % self._t_patch_size == 0 and self._input_size % self._patch_size == 0
        self._N = int(self._repeat_aug)
        self._L = int((self._num_frames // self._t_patch_size) * (
                self._input_size // self._patch_size) ** 2)  # total num of patches ()

        # 选择mask方法
        # if mask_strategy == 'DFDH':  # Depth and frame difference prompts（our methods）
        #     self._masking = self.DFDH
        # elif mask_strategy == 'FDH':  # Depth and frame difference prompts（our methods）
        #     self._masking = self.FDH
        # elif mask_strategy == 'random':
        #     self._masking = self.random_masking
        # else:
        #     raise ValueError(f"mask_strategy {mask_strategy} not defined!")
        self.logger.info(f"mask_strategy: {mask_strategy}")

        MaskMapGenerator.__init__(self)

    def __getitem__(self, index):
        frames_list, frames_gray_list, frames_depth_gray_list = [], [], []
        success_sampled = False
        while not success_sampled:  # 当文件存在且可以打开，但在使用时仍因为不明原因报错时，回到此处重新执行
            # 获取视频文件
            cap, sample, index = self._get_video_cap(index)
            # 获取视频采样和帧差图
            frames_list, frames_gray_list, start_idx, end_idx, success_sampled \
                = self._get_snippietsAndframeDif(cap, index)
            if not success_sampled: continue
            # 获取深度图
            if self.use_depth:
                frames_depth_gray_list, success_sampled \
                    = self._get_snippiets_depth(index, start_idx, end_idx, frames_gray_list)
                if not success_sampled: continue

        frames = np.array(frames_list)
        frames_gray = np.array(frames_gray_list)
        mask_prob_frame = self.mask_probability_single(frames_gray)
        if self.use_depth:
            frames_depth_gray = np.array(frames_depth_gray_list)
            t, h, w = frames_depth_gray.shape  # (16, 356, 454)
            frames_depth_gray = frames_depth_gray.reshape((1, t, h, w))
            mask_prob_depth = self.mask_probability_depth(frames_depth_gray)
        else:
            mask_prob_depth = np.ones_like(mask_prob_frame)
        mask, ids_restore, ids_keep, ids_mask = self.mask_probability_generator(mask_prob_frame, mask_prob_depth)

        return frames, mask, ids_restore, ids_keep, ids_mask

    def __len__(self):
        return len(self._path_to_videos)


class VideoMaeDataset_difToken(VideoDataset, MaskMapGenerator):
    def __init__(
            self,
            mode,
            path_to_data_dir,
            logger,

            # decoding setting
            sampling_rate=4,
            num_frames=16,
            input_size=224,

            # other parameters
            use_offset_sampling=True,
            num_retries=10,

            # pretrain augmentation
            repeat_aug=1,

            # VideoMaeDataset set
            # random mask
            mask_strategy='random',  # random / DFDH / FDH
            t_patch_size=2,
            patch_size=16,  # the size of patch is 16*16*2
            mask_ratio=0.9,
    ):
        VideoDataset.__init__(
            self,
            mode=mode,
            path_to_data_dir=path_to_data_dir,
            logger=logger,
            sampling_rate=sampling_rate,
            num_frames=num_frames,
            repeat_aug=repeat_aug,
            num_retries=num_retries,
            input_size=input_size,
            use_offset_sampling=use_offset_sampling,
        )
        # For training or validation mode, one single clip is sampled from every video.
        # For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every video.
        # For every clip, NUM_SPATIAL_CROPS is cropped spatially from the frames.
        self._num_clips = 1

        self.logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        if mask_strategy == 'DFDH':
            self._construct_loader_depth()
            self.use_depth = True
        else:
            self.use_depth = False
        self._t_patch_size = t_patch_size
        self._patch_size = patch_size
        self._mask_ratio = mask_ratio
        self._patch_num = int(self._input_size // self._patch_size)

        assert self._num_frames % self._t_patch_size == 0 and self._input_size % self._patch_size == 0
        self._N = int(self._repeat_aug)
        self._L = int((self._num_frames // self._t_patch_size) * (
                self._input_size // self._patch_size) ** 2)  # total num of patches ()

        # 选择mask方法
        # if mask_strategy == 'DFDH':  # Depth and frame difference prompts（our methods）
        #     self._masking = self.DFDH
        # elif mask_strategy == 'FDH':  # Depth and frame difference prompts（our methods）
        #     self._masking = self.FDH
        # elif mask_strategy == 'random':
        #     self._masking = self.random_masking
        # else:
        #     raise ValueError(f"mask_strategy {mask_strategy} not defined!")
        self.logger.info(f"mask_strategy: {mask_strategy}")

        MaskMapGenerator.__init__(self)

    def __getitem__(self, index):
        frames_list, frames_gray_list, frames_depth_gray_list = [], [], []
        success_sampled = False
        while not success_sampled:  # 当文件存在且可以打开，但在使用时仍因为不明原因报错时，回到此处重新执行
            # 获取视频文件
            cap, sample, index = self._get_video_cap(index)
            # 获取视频采样和帧差图
            frames_list, frames_gray_list, start_idx, end_idx, success_sampled \
                = self._get_snippietsAndframeDif(cap, index)
            if not success_sampled: continue
            # 获取深度图
            if self.use_depth:
                frames_depth_gray_list, success_sampled \
                    = self._get_snippiets_depth(index, start_idx, end_idx, frames_gray_list)
                if not success_sampled: continue

        frames = np.array(frames_list)
        frames_gray = np.array(frames_gray_list)
        mask_prob_frame = self.mask_probability_single(frames_gray)
        if self.use_depth:
            frames_depth_gray = np.array(frames_depth_gray_list)
            t, h, w = frames_depth_gray.shape  # (16, 356, 454)
            frames_depth_gray = frames_depth_gray.reshape((1, t, h, w))
            mask_prob_depth = self.mask_probability_depth(frames_depth_gray)
        else:
            mask_prob_depth = np.ones_like(mask_prob_frame)
        mask, ids_restore, ids_keep, ids_mask = self.mask_probability_generator(mask_prob_frame, mask_prob_depth)

        return frames, mask, ids_restore, ids_keep, ids_mask

    def __len__(self):
        return len(self._path_to_videos)


class VideoClsDataset(VideoDataset, MaskMapGenerator):
    def __init__(
            self,
            mode,
            path_to_data_dir,
            logger,

            # decoding setting
            sampling_rate=4,
            num_frames=16,
            input_size=224,

            # pretrain augmentation
            repeat_aug=1,

            # other parameters
            use_offset_sampling=True,
            num_retries=10,

            # VideoClsDataset set
            # test setting, multi crops
            test_num_ensemble_views=10,
            test_num_spatial_crops=3,
    ):
        VideoDataset.__init__(
            self,
            mode=mode,
            path_to_data_dir=path_to_data_dir,
            logger=logger,
            sampling_rate=sampling_rate,
            num_frames=num_frames,
            repeat_aug=repeat_aug,
            num_retries=num_retries,
            input_size=input_size,
            use_offset_sampling=use_offset_sampling,
        )

        self._test_num_ensemble_views = test_num_ensemble_views
        self._test_num_spatial_crops = test_num_spatial_crops

        # For training or validation mode, one single clip is sampled from every video.
        # For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every video.
        # For every clip, NUM_SPATIAL_CROPS is cropped spatially from the frames.
        if self.mode in ["finetune", "val"]:
            self._num_clips = 1
            self._test_num_ensemble_views = 1
        elif self.mode in ["test"]:
            self._num_clips = test_num_ensemble_views * test_num_spatial_crops

        self._construct_loader()

    def __getitem__(self, index):

        success_sampled = False
        while not success_sampled:  # 当文件存在且可以打开，但在使用时仍应为不明原因报错时，回到此处重新执行
            # 获取视频文件
            cap, sample, index = self._get_video_cap(index)
            # 视频采样
            frame_list, label_list, success_sampled = self._get_snippiets(cap, index)

        if self.mode in ["finetune", "val"]:
            return frame_list, label_list
        elif self.mode == "test":
            return frame_list, label_list, sample.split("/")[-1].rstrip(".mp4"), self._spatial_temporal_idx[index]

    def __len__(self):
        return len(self._path_to_videos)


def create_dataset(args, mode="pretrain", shuffle=True):
    # init Dataset and data_transform
    if mode == "pretrain":
        video_dataset = VideoMaeDataset(
            mode=mode,
            path_to_data_dir=args.path_to_data_dir,
            logger=args.logger,
            sampling_rate=args.sampling_rate,
            num_frames=args.num_frames,
            repeat_aug=args.repeat_aug,

            # random mask相关参数
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
            mask_ratio=args.mask_ratio,
        )
        column_names = ["video", "mask", "ids_restore", "ids_keep", "ids_mask"]

        data_transform = transforms.Compose([  # r, t, h, w, c
            vision.Rescale(rescale=1.0 / 255.0, shift=0),
            vision.Normalize(mean=args.mean, std=args.std),
            video_utils.TransposeReshape(args.repeat_aug),  # h, w, c*r*t
            vision.RandomResizedCrop(size=args.input_size, scale=args.jitter_scales_relative,
                                     ratio=args.jitter_aspect_relative),
            vision.RandomHorizontalFlip(args.train_random_horizontal_flip),
            vision.HWC2CHW(),
        ])

    else:
        video_dataset = VideoClsDataset(
            mode=mode,
            path_to_data_dir=args.path_to_data_dir,
            logger=args.logger,
            sampling_rate=args.sampling_rate,
            num_frames=args.num_frames,
            test_num_ensemble_views=args.test_num_ensemble_views,
            test_num_spatial_crops=args.test_num_spatial_crops,
            repeat_aug=args.repeat_aug,
            use_offset_sampling=args.use_offset_sampling
        )
        if mode == "test":
            column_names = ["video", "label", "sample", "index"]
        elif mode in ["finetune", "val"]:
            column_names = ["video", "label"]

        if mode == "finetune":
            data_transform = transforms.Compose([
                video_utils.RandAugmentTransform(args.input_size, args.auto_augment, args.interpolation),
                vision.Rescale(rescale=1.0 / 255.0, shift=0),
                video_utils.NumpyNormalize(mean=args.mean, std=args.std),
                video_utils.TransposeReshape(args.repeat_aug),
                vision.RandomResizedCrop(size=args.input_size, scale=args.jitter_scales_relative,
                                         ratio=args.jitter_aspect_relative),
                vision.RandomHorizontalFlip(args.train_random_horizontal_flip),
                vision.HWC2CHW(),
                video_utils.ReshapeTranspose(args.repeat_aug, args.num_frames),
                # 随机擦除
                video_utils.RandomErase(args.reprob, mode=args.remode, max_count=args.recount, num_splits=args.recount)
            ])
        elif mode == "val":
            data_transform = transforms.Compose([
                vision.Rescale(rescale=1.0 / 255.0, shift=0),
                video_utils.NumpyNormalize(mean=args.mean, std=args.std),
                video_utils.TransposeReshape(args.repeat_aug),
                vision.Resize(size=args.short_side_size, interpolation=Inter.BILINEAR),
                vision.CenterCrop(size=args.train_crop_size),
                vision.HWC2CHW(),
                video_utils.ReshapeTranspose(args.repeat_aug, args.num_frames)
            ])
        elif mode == "test":
            data_transform = transforms.Compose([
                video_utils.TestAUG(rescale=1.0 / 255.0,
                                    shift=0,
                                    mean=args.mean,
                                    std=args.std,
                                    repeat_aug=args.repeat_aug,
                                    short_side_size=args.short_side_size,
                                    interpolation=Inter.BILINEAR,
                                    cropsize=args.train_crop_size,
                                    test_num_spatial_crops=args.test_num_spatial_crops,
                                    num_frames=args.num_frames)
            ])

    args.logger.info("jitter_aspect_relative {} jitter_scales_relative {}".format(args.jitter_aspect_relative,
                                                                                  args.jitter_scales_relative))
    cores = multiprocessing.cpu_count()
    args.logger.info("cores: {}".format(cores))
    if mode in ["pretrain", "finetune"]:
        num_parallel_workers = max(min(args.batch_size * args.repeat_aug * 2, int(cores / args.device_num)), 32)
    else:
        num_parallel_workers = 8
    if args.device_target in ['cpu', 'CPU']:
        num_parallel_workers = 8

    args.logger.info("num_parallel_workers: {}".format(num_parallel_workers))
    sampler = DistributedSampler(args.device_num, args.local_rank, shuffle=shuffle)

    data = dataset.GeneratorDataset(
        source=video_dataset,
        column_names=column_names,
        num_parallel_workers=8 if mode in ["finetune", "pretrain"] else 1,
        python_multiprocessing=False,
        sampler=sampler
    )

    # Data transform
    if mode in ["finetune", "pretrain", "val"]:
        data = data.map(data_transform,
                        input_columns="video",
                        num_parallel_workers=num_parallel_workers)
    elif mode == "test":
        data = data.map(data_transform,
                        input_columns=["video", "label", "sample", "index"],
                        output_columns=["video", "label", "sample", "index"],
                        column_order=["video", "label", "sample", "index"],
                        num_parallel_workers=num_parallel_workers)
    data = data.batch(
        args.batch_size,
        drop_remainder=True,  # 全局默认线程数：8
    )

    # finetune MixUp
    if mode == "finetune":
        if args.mixup_active:
            args.logger.info("MixUp Opended!")
            data = data.map(
                Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                      prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                      label_smoothing=args.label_smoothing, num_classes=args.num_classes),
                input_columns=["video", "label"]
            )
        else:
            args.logger.info("MixUp Closed!")

    return data
