import multiprocessing
import os

import cv2
import mindspore as ms
import numpy as np
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
            mask_strategy='random',  # 'tube' or 'random
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

        # random mask相关
        if mask_strategy == 'tube':
            self._masking = self.tube_masking
        elif mask_strategy == 'random':
            self._masking = self.random_masking
        else:
            raise ValueError(f"mask_strategy {mask_strategy} not defined!")
        self.logger.info(f"mask_strategy: {mask_strategy}")

        self._t_patch_size = t_patch_size
        self._patch_size = patch_size
        self._mask_ratio = mask_ratio

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
            "pretrain_val":"val",
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

    def __getitem__(self, index):
        # -1 indicates random sampling.
        temporal_sample_index = -1
        spatial_sample_index = -1

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

            frames_list = []
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
                # video_start_frame = int(0) # 前向对齐用
                # video_end_frame = int(63)

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
                    frames_list.append(frames)
                    if len(frames_list) == self._repeat_aug:
                        success_sampled = True
                except:
                    self.logger.warning(f"temporal sampling failed,file:{sample}")
                    index = np.random.randint(0, len(self._path_to_videos) - 1)
                    sample = self._path_to_videos[index]
                    cap = cv2.VideoCapture(sample)

        cap.release()

        # random mask
        frames = np.array(frames_list)
        mask, ids_restore, ids_keep, ids_mask = self._masking(self._N, self._L)
        return frames, mask, ids_restore, ids_keep, ids_mask

    def __len__(self):
        return len(self._path_to_videos)
        # return 2560

    def random_masking(self, N, L):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        len_keep = int(L * (1 - self._mask_ratio))

        noise = np.random.rand(N, L)  # np.random.rand(L)

        # sort noise for each sample
        ids_shuffle = np.argsort(noise, axis=-1).astype(np.int32)
        ids_restore = np.argsort(ids_shuffle, axis=-1).astype(np.int32)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]

        mask = np.ones((N, L), dtype=np.float32)
        mask[:, :len_keep] = 0.0

        for i in range(N):
            mask[i] = mask[i, ids_restore[i]]

        return mask, ids_restore, ids_keep, ids_mask


    def tube_masking(self, N, L):
        """
        对每个样本进行管道掩码，通过对随机噪声进行复制拼贴后进行argsort来对每个样本进行per-sample shuffling。
        x: [N, L, D]，表示序列数据，其中N表示样本数量，L表示序列长度，D表示特征维度
        """
        # 计算每帧的图像patch/token数目
        num_patches_per_frame = (self._input_size / self._patch_size) ** 2
        L_per_frame = num_patches_per_frame

        # 计算每帧保留的patch/token数目
        len_keep_per_frame = int(L_per_frame * (1 - self._mask_ratio))
        t_clip = self._num_clips / self._clip_size

        # 计算整个序列保留的补丁数目
        len_keep = len_keep_per_frame * t_clip

        # 生成随机噪声并进行复制得到所需的数量,复制的作用是实现tube mask效果
        noise = np.random.rand(N, L_per_frame)
        noise = np.tile(noise, (1, t_clip))

        # 对随机噪声进行argsort操作，得到shuffle后的索引
        ids_shuffle = np.argsort(noise, axis=-1).astype(np.int32)
        # 对shuffle后的索引进行argsort操作，得到还原后的索引
        ids_restore = np.argsort(ids_shuffle, axis=-1).astype(np.int32)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]

        mask = np.ones((N, L), dtype=np.float32)
        mask[:, :len_keep] = 0.0

        for i in range(N):
            mask[i] = mask[i, ids_restore[i]]

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
        while not success_sampled:  #当文件存在且被cv2认为可以打开，但在使用时仍应为不明原因报错时，回到此处重新执行
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
                    if len(label_list) == self._repeat_aug:
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
    assert mode in ["pretrain","pretrain_val", "finetune"],r"dataset mode should in [pretrain,pretrain_val,finetune]"
    if "pretrain" in mode:   # pretrain/pretrain_val
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
            vision.HWC2CHW(), # c*r*t,h,w
        ])
        # data_transform = transforms.Compose([ # r, t, h, w, c
        #     vision.Rescale(rescale=1.0/255.0, shift=0),
        #     vision.Normalize(mean=args.mean, std=args.std),
        #     video_utils.TransposeReshape(args.repeat_aug),
        #     video_utils.Crop(size=args.input_size),
        #     vision.HWC2CHW(),
        # ])
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
    if mode in ["pretrain","pretrain_val", "finetune"]:
        num_parallel_workers = max(min(args.batch_size * args.repeat_aug * 2, int(cores / args.device_num)), 32)
    else:
        num_parallel_workers = max(min(args.batch_size * args.repeat_aug * 2, int(cores / args.device_num)), 32)
    if args.device_target in ['cpu', 'CPU']:
        num_parallel_workers = 8


    args.logger.info("num_parallel_workers: {}".format(num_parallel_workers))
    sampler = DistributedSampler(args.device_num, args.local_rank, shuffle=shuffle)

    data = dataset.GeneratorDataset(
        source=video_dataset,
        column_names=column_names,
        num_parallel_workers=8 if mode in ["finetune", "pretrain","pretrain_val"] else 1,
        python_multiprocessing=False,
        sampler=sampler
    )

    if mode in ["finetune", "pretrain","pretrain_val"]:
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
        drop_remainder=True,
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

    # if mode != "pretrain":
    #     data=data.map(
    #             video_utils.TransposeReshape_2Bcthw(mode=mode, batch_size=args.batch_size, repeat_aug=args.repeat_aug),
    #             input_columns=["video", "label"]
    #         )
    # else:
    #     data=data.map(
    #             video_utils.TransposeReshape_2Bcthw(mode=mode, batch_size=args.batch_size, repeat_aug=args.repeat_aug),
    #             input_columns=["video", "mask", "ids_restore", "ids_keep"]
    #         )

    return data
