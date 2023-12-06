import multiprocessing
import os

import cv2
import mindspore as ms
import numpy as np
import random
import mindspore.common.dtype as mstype
import mindspore.ops as ops

from mindspore import Tensor
from mindspore import dataset
from mindspore.dataset import (DistributedSampler, SequentialSampler,
                               transforms, vision)
from mindspore.dataset.vision import Inter

from transforms.mixup import Mixup
from utils_mae import video_utils


class VideoMaeDataset:
    def __init__(
            self,
            mode,
            path_to_data_dir,
            logger,

            sampling_rate=4,
            num_frames=16,
            input_size=224,

            # other parameters
            use_offset_sampling=True,
            # inverse_uniform_sampling=False,
            num_retries=10,

            # pretrain augmentation
            repeat_aug=1,

            # random mask
            mask_strategy='tube',  # 'tube' or 'random
            t_patch_size=2,
            patch_size=16,
            mask_ratio=0.9,
    ):
        self.mode = mode
        self.logger = logger

        self._repeat_aug = repeat_aug
        self._num_retries = num_retries
        self._path_to_data_dir = path_to_data_dir

        self._input_size = input_size

        self._sampling_rate = sampling_rate
        self._num_frames = num_frames
        self._clip_size = self._sampling_rate * self._num_frames

        # self._inverse_uniform_sampling = inverse_uniform_sampling
        self._use_offset_sampling = use_offset_sampling
        self.logger.info("Constructing Videos {}...".format(mode))

        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        self._num_clips = 1

        self.logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

        # # random mask相关
        # if mask_strategy == 'tube':
        #     self._masking = self.tube_masking
        # elif mask_strategy == 'random':
        #     self._masking = self.random_masking
        # else:
        #     raise ValueError(f"mask_strategy {mask_strategy} not defined!")
        # self.logger.info(f"mask_strategy: {mask_strategy}")

        self._t_patch_size = t_patch_size
        self._patch_size = patch_size
        self._mask_ratio = mask_ratio
        self._path_num = int(self._input_size // self._patch_size)

        assert self._num_frames % self._t_patch_size == 0 and self._input_size % self._patch_size == 0
        self._N = int(self._repeat_aug)
        self._L = int((self._num_frames // self._t_patch_size) * (self._input_size // self._patch_size) ** 2)

    def _construct_loader(self):
        # 拿到所有视频的路径
        """
        Construct the video loader.
        """
        csv_file_name = {
            "pretrain": "train",
            "finetune": "train",
            "val": "val",
            "test": "test",
        }
        path_to_file = os.path.join(
            self._path_to_data_dir,
            "{}.csv".format(csv_file_name[self.mode]),
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)

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

        assert len(self._path_to_videos) > 0, "Failed to load Kinetics from {}".format(path_to_file)

        self.logger.info("Constructing video dataloader (size: {}) from {}".format(
            len(self._path_to_videos), path_to_file)
        )

        # depth
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

    def __getitem__(self, index):
        # -1 indicates random sampling.
        temporal_sample_index = -1
        spatial_sample_index = -1

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        sample = self._path_to_videos[index]
        sample_depth = self._path_to_videos_depth[index]

        cap = cv2.VideoCapture(sample)
        success_sampled = False
        while not success_sampled:  # 当文件存在且被cv2认为可以打开，但在使用时仍应为不明原因报错时，回到此处重新执行
            for retry in range(self._num_retries):  # 视频打开失败，重新随机抓取视频
                if cap.isOpened():
                    break
                self.logger.warning("视频打开失败,video id: {}, retries: {}, path: {}"
                                    .format(index, retry, sample))
                if not os.path.exists(sample):
                    self.logger.warning("文件{}不存在".format(sample))
                index = np.random.randint(0, len(self._path_to_videos) - 1)
                sample = self._path_to_videos[index]
                cap = cv2.VideoCapture(sample)

            total_frames = cap.get(7)  # 视频总帧数

            frames_list = []
            frames_gray_list = []
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
                video_end_frame = int(end_idx)

                cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

                frames = []
                frames_gray = []
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
                except:
                    self.logger.warning(f"temporal sampling failed,file:{sample}")
                    index = np.random.randint(0, len(self._path_to_videos) - 1)
                    sample = self._path_to_videos[index]
                    cap = cv2.VideoCapture(sample)

        cap.release()

        # depth
        cap_d = cv2.VideoCapture(sample_depth)
        success_sampled = False

        # 对应原始视频64帧的  7帧深度图的起始id
        start_idx_d = int(start_idx / 10)
        end_idx_d = int(end_idx / 10)
        clip_size_d = int(end_idx_d - start_idx_d) + 1
        while not success_sampled:
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
                    ids = np.random.randint(0, len(self._path_to_videos_depth) - 1)
                    sample_depth = self._path_to_videos_depth[ids]
                    cap_d = cv2.VideoCapture(sample_depth)

        cap_d.release()

        frames = np.array(frames_list)
        frames_gray = np.array(frames_gray_list)
        frames_depth_gray = np.array(frames_depth_gray_list)
        t, h, w = frames_depth_gray.shape  # (16, 356, 454)
        frames_depth_gray = frames_depth_gray.reshape((1, t, h, w))

        mask_prob_frame = self.mask_probability_single(frames_gray)
        mask_prob_depth = self.mask_probability_depth(frames_depth_gray)
        mask, ids_restore, ids_keep, ids_mask = self.mask_probability_generator(mask_prob_frame, mask_prob_depth)

        # loss_weight = loss_weight[ids_mask]
        # loss_weight = Tensor(loss_weight, mstype.float32)

        return frames, mask, ids_restore, ids_keep, ids_mask

    def __len__(self):
        return len(self._path_to_videos)

    def mask_probability_single(self, frames_gray):
        # 单帧差法生成mask 概率图
        r, t, h, w, = frames_gray.shape  # 1, 16, 320, 426)
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

        # 原始宽高的帧差resize到（224, 224)
        resize_op = vision.RandomResizedCrop(size=224, scale=(0.5, 1.0), ratio=(0.75, 1.333))
        diff_list_done = []
        for i in range(t):
            diff_list1 = resize_op(diff_list_0[i])
            diff_list_done.append(diff_list1)
        diff_list_done = np.array(diff_list_done)  # shape:(16, 224, 224)

        # patch:16*16 帧差下采样 (16, 224, 224)--->(16, 14, 14)
        patch_prob = []
        threshold = int(10)
        for diff in diff_list_done:
            patch = cv2.resize(diff, (self._path_num, self._path_num), interpolation=cv2.INTER_LINEAR)  # 下采样

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

    def mask_probability_depth(self, frames_depth_gray):
        r, t, h, w, = frames_depth_gray.shape  # 1, 16, 320, 426)
        frames_depth_gray = frames_depth_gray.reshape(t, h, w)  # (16, 320, 426)

        # ori -->(16, 250, 490) --->(16, 224, 224)
        resize_op = vision.RandomResizedCrop(size=224, scale=(0.5, 1.0), ratio=(0.75, 1.333))
        depth_resize = []
        for i in range(t):
            resize = resize_op(frames_depth_gray[i])
            depth_resize.append(resize)

        depth_resize = np.array(depth_resize)  # (16, 224, 224)

        # patch:16*16 帧差下采样 (25, 224, 224)--->(25, 14, 14)
        depth_patch_prob = []
        # threshold = int(10)
        for frame in depth_resize:
            patch = cv2.resize(frame, (14, 14), interpolation=cv2.INTER_LINEAR)  # 下采样
            # patch 为0值 随机设置（1，5）之间的值
            patch = np.where(patch == 0, np.random.randint(1, 5), patch)
            depth_patch_prob.append(patch)

        depth_patch_prob = np.array(depth_patch_prob)  # (16, 14, 14)

        #         weight = []
        #         loss_weight = []
        #         # loss_weight
        #         for patch in depth_patch_prob:
        #             weight_1 = np.where(((patch >= 0) & (patch <=50)), 1.6, patch)
        #             weight_2 = np.where(((patch > 50) & (patch <=100)), 1.2, weight_1)
        #             weight_3 = np.where(((patch > 100) & (patch <=150)), 0.8, weight_2)
        #             weight_4 = np.where(patch >= 150, 0.4, weight_3)
        #             weight.append(weight_4)

        #         weight = np.array(weight)   #  (16, 14, 14)

        #         for i in range(8):
        #             loss_weight.append(weight[2*i])

        #         loss_weight = np.array(loss_weight).flatten()   # (8, 14, 14)

        # 生成概率图
        mask_prob_depth = []
        for patch in depth_patch_prob:
            prob = patch / np.sum(patch)
            mask_prob_depth.append(prob)

        mask_prob_depth = np.array(mask_prob_depth)  # (16, 14, 14)

        return mask_prob_depth

    def mask_probability_generator(self, mask_prob_frame, mask_prob_depth):
        mask_prob = []
        prob_mul = mask_prob_frame * mask_prob_depth
        for p in prob_mul:
            prob = p / np.sum(p)
            mask_prob.append(prob)

        mask_prob = np.array(mask_prob)

        patch_num = int(self._path_num ** 2)  # 196

        # 0：保留   1：mask
        # num_elements = int(self._mask_ratio * patch_num)
        num_elements = int(self._mask_ratio * patch_num)  # 75%: mask-176  save-20
        mask, ids_mask, ids_keep, ids_restore = [[] for i in range(4)]

        # mask_prob:(16, 14, 14)
        # 两帧采一帧
        j = 0
        for i in range(8):
            num_base = i * patch_num

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


class VideoClsDataset:
    def __init__(
            self,
            mode,
            path_to_data_dir,

            logger,

            # decoding setting
            sampling_rate=4,
            num_frames=16,

            # test setting, multi crops
            test_num_ensemble_views=10,
            test_num_spatial_crops=3,

            # pretrain augmentation
            repeat_aug=1,

            # other parameters
            use_offset_sampling=True,
            num_retries=10,
    ):
        self.mode = mode
        self.logger = logger
        self._path_to_data_dir = path_to_data_dir

        self._sampling_rate = sampling_rate
        self._num_frames = num_frames
        self._clip_size = self._sampling_rate * self._num_frames

        self._test_num_ensemble_views = test_num_ensemble_views
        self._test_num_spatial_crops = test_num_spatial_crops

        self._repeat_aug = repeat_aug

        self._use_offset_sampling = use_offset_sampling
        self._num_retries = num_retries

        self.logger.info("Constructing Videos {}...".format(mode))

        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["finetune", "val"]:
            self._num_clips = 1
            self._test_num_ensemble_views = 1
        elif self.mode in ["test"]:
            self._num_clips = test_num_ensemble_views * test_num_spatial_crops

        self._construct_loader()

    def _construct_loader(self):
        # 拿到所有视频的路径
        """
        Construct the video loader.
        """
        csv_file_name = {
            "pretrain": "train",
            "finetune": "train",
            "val": "val",
            "test": "test",
        }
        path_to_file = os.path.join(
            self._path_to_data_dir,
            "{}.csv".format(csv_file_name[self.mode]),
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 2
                path, label = path_label.split()
                for idx in range(self._num_clips):
                    self._path_to_videos.append(os.path.join(path))
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)

        assert len(self._path_to_videos) > 0, "Failed to load Kinetics from {}".format(path_to_file)

        self.logger.info("Constructing video dataloader (size: {}) from {}".format(
            len(self._path_to_videos), path_to_file)
        )

    def __getitem__(self, index):

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        sample = self._path_to_videos[index]
        cap = cv2.VideoCapture(sample)
        success_sampled = False
        while not success_sampled:  # 当文件存在且被cv2认为可以打开，但在使用时仍应为不明原因报错时，回到此处重新执行
            for retry in range(self._num_retries):  # 视频打开失败，重新随机抓取视频
                if cap.isOpened():
                    break
                self.logger.warning("视频打开失败,video id: {}, retries: {}, path: {}"
                                    .format(index, retry, sample))
                if not os.path.exists(sample):
                    self.logger.warning("文件{}不存在".format(sample))
                index = np.random.randint(0, len(self._path_to_videos) - 1)
                sample = self._path_to_videos[index]
                cap = cv2.VideoCapture(sample)

            total_frames = cap.get(7)  # 视频总帧数

            if self.mode in "finetune":
                temporal_sample_index = -1
                spatial_sample_index = -1
            elif self.mode == "val":
                temporal_sample_index = 0
                spatial_sample_index = -1
            elif self.mode == "test":
                temporal_sample_index = self._spatial_temporal_idx[index] // self._test_num_spatial_crops
                spatial_sample_index = self._spatial_temporal_idx[index] % self._test_num_spatial_crops \
                    if self._test_num_spatial_crops > 1 else 1
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
                    self.logger.warning(f"temporal sampling failed,file:{sample}")
                    index = np.random.randint(0, len(self._path_to_videos) - 1)
                    sample = self._path_to_videos[index]
                    cap = cv2.VideoCapture(sample)

        frame_list = np.array(frame_list)
        label_list = np.array(label_list, dtype=np.int32)

        cap.release()

        if self.mode in ["finetune", "val"]:
            return frame_list, label_list
        elif self.mode == "test":
            # print(sample.split("/")[-1].rstrip(".mp4"))   #目前只设计了针对k400的版本
            return frame_list, label_list, sample.split("/")[-1].rstrip(".mp4"), self._spatial_temporal_idx[index]

    def __len__(self):
        return len(self._path_to_videos)
        # return 1280


def create_dataset(args, mode="pretrain", shuffle=True):
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
                # mean=[args.mean[0]]*(3*args.num_frames*args.repeat_aug), std=[args.std[0]]*(3*args.num_frames*args.repeat_aug)
                video_utils.TransposeReshape(args.repeat_aug),
                vision.RandomResizedCrop(size=args.input_size, scale=args.jitter_scales_relative,
                                         ratio=args.jitter_aspect_relative),
                vision.RandomHorizontalFlip(args.train_random_horizontal_flip),
                vision.HWC2CHW(),
                video_utils.ReshapeTranspose(args.repeat_aug, args.num_frames),
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

    if mode in ["finetune", "pretrain"]:
        data = data.map(data_transform,
                        input_columns="video",
                        num_parallel_workers=num_parallel_workers)
    elif mode == "val":
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

    if mode == "finetune" and args.mixup_active:
        args.logger.info("MixUp Opended!")
        data = data.map(
            Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                  prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                  label_smoothing=args.label_smoothing, num_classes=args.num_classes),
            input_columns=["video", "label"]
        )
    elif mode == "finetune":
        args.logger.info("MixUp Closed!")

    return data
