import math
import random

import mindspore as ms
import numpy as np
from mindspore.dataset import vision
from mindspore.dataset.vision import Inter
from PIL import Image

from transforms.auto_augment import rand_augment_transform
from transforms.random_erasing import RandomErasing


def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR


class Compose:
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, frames):
        for t in self._transforms:
            frames = t(frames)
        return frames

    def append(self, transform):
        self._transforms.append(transform)


class RandAugmentTransform:
    def __init__(self, input_size, auto_augment, interpolation='bilinear'):
        self.auto_augment = auto_augment
        if isinstance(input_size, tuple):
            img_size = input_size[-2:]
        else:
            img_size = input_size
        if auto_augment:
            assert isinstance(auto_augment, str)
            if isinstance(img_size, tuple):
                img_size_min = min(img_size)
            else:
                img_size_min = img_size
            aa_params = {"translate_const": int(img_size_min * 0.45)}
            if interpolation and interpolation != "random":
                aa_params["interpolation"] = _pil_interp(interpolation)
            if auto_augment.startswith("rand"):
                self.rand_augment_transform = rand_augment_transform(auto_augment, aa_params)

    def __call__(self, frames):
        r, t, h, w, c = frames.shape
        if self.auto_augment:
            new_frames = []
            for i in range(r):
                buffer = frames[i]
                buffer = [Image.fromarray(img) for img in buffer]
                buffer = self.rand_augment_transform(buffer)

                buffer = [np.array(img) for img in buffer]
                buffer = np.stack(buffer)  # T H W C
                new_frames.append(buffer)
            new_frames = np.stack(new_frames)  # R T H W C
            return new_frames
        else:
            return frames


class RandomErase:
    def __init__(self, reprob=0.5, mode='const', max_count=None, num_splits=0):
        self.rand_erase = True if reprob > 0 else False
        if self.rand_erase:
            self.erase_transform = RandomErasing(
                probability=reprob,
                mode=mode,
                max_count=max_count,
                num_splits=num_splits,
            )

    def __call__(self, frames):
        r, c, t, h, w = frames.shape
        if self.rand_erase:
            new_frames = []
            for frame in frames:  # c, t, h, w
                frame = frame.transpose(1, 0, 2, 3)  # t, c, h, w
                frame = self.erase_transform(frame)
                frame = frame.transpose(1, 0, 2, 3)  # c, t, h, w
                new_frames.append(frame)
            new_frames = np.stack(new_frames)
            return new_frames
        else:
            return frames


class Rescale:
    def __init__(self, scale=1.0 / 255.0, shift=0):
        self._scale = scale
        self._shift = shift
        self.rescale = vision.Rescale(rescale=self._scale, shift=self._shift)

    def __call__(self, frames):
        frames = self.rescale(frames)
        return frames


class Normalize:
    def __init__(self, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)):
        self._mean = mean
        self._std = std
        self.normalize = vision.Normalize(mean=self._mean, std=self._std)

    def __call__(self, frames):
        frames = self.normalize(frames)
        return frames


class Resize:
    def __init__(self, size, interpolation=Inter.BILINEAR):
        self._size = size
        self._interpolation = interpolation
        self.resize = vision.Resize(self._size, self._interpolation)

    def __call__(self, frames):
        frames = self.resize(frames)
        return frames


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self._size = size
        self.center_crop = vision.CenterCrop(self._size)

    def __call__(self, frames):
        T, H, W, C = frames.shape
        new_frames = []
        for i in range(T):  # mindspore1.9版本仅支持输入[H, W]或[H, W, C]
            new_frames.append(self.center_crop(frames[i]))
        new_frames = np.array(new_frames)
        return new_frames


class CenterCrop_Gray:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self._size = size
        self.center_crop = vision.CenterCrop(self._size)

    def __call__(self, frames):
        T, H, W = frames.shape
        new_frames = []
        for i in range(T):  # mindspore1.9版本仅支持输入[H, W]或[H, W, C]
            new_frames.append(self.center_crop(frames[i]))
        new_frames = np.array(new_frames)
        return new_frames


class RandomResizedCrop:
    def __init__(self, size=244, scale=(0.5, 1.0), ratio=(0.75, 1.333)):
        self._scale = scale
        self._ratio = ratio
        if isinstance(size, int):
            size = (size, size)
        self._size = size
        self.random_resized_crop = vision.RandomResizedCrop(size=self._size, scale=self._scale, ratio=self._ratio)

    def __call__(self, frames):
        T, H, W, C = frames.shape
        new_frames = []
        for i in range(T):  # mindspore1.9版本仅支持输入[H, W]或[H, W, C]
            new_frames.append(self.random_resized_crop(frames[i]))
        new_frames = np.array(new_frames)
        return new_frames


class RandomResizedCrop_self:
    def __init__(self, size=224, scale=(0.5, 1.0), ratio=(0.75, 1.333)):
        self._scale = scale
        self._ratio = ratio
        if isinstance(size, int):
            size = (size, size)
        self._size = size
        self.resize = vision.Resize(self._size)

    def __call__(self, frames):
        T, H, W, C = frames.shape
        start_x, start_y, target_h, target_w = _get_param_spatial_crop(self._scale, self._ratio, H, W)
        crop = vision.Crop((start_x, start_y), (target_h, target_w))

        new_frames = []
        for i in range(T):
            new_frames.append(self.resize(crop(frames[i])))
        new_frames = np.array(new_frames)
        return new_frames


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self._prob = prob
        self.horizontal_flip = vision.HorizontalFlip()

    def __call__(self, frames):
        T, _, _, _ = frames.shape
        if np.random.uniform() < self._prob:
            for i in range(T):
                frames[i] = self.horizontal_flip(frames[i])
        return frames


class TransposeReshape:
    def __init__(self, repeat_aug=1):
        self._repeat_aug = repeat_aug

    def __call__(self, frames):
        r, t, h, w, c = frames.shape
        frames = frames.transpose(2, 3, 0, 4, 1)  # h, w, r, c, t
        frames = frames.reshape(h, w, r * c * t)
        return frames


class ReshapeTranspose:
    def __init__(self, repeat_aug=1, num_frames=16):
        self._repeat_aug = repeat_aug
        self._num_frames = num_frames

    def __call__(self, frames):
        c, h, w = frames.shape
        frames = frames.reshape(self._repeat_aug, 3, self._num_frames, h, w)  # r, c, t, h, w
        # frames = frames.transpose(1, 0, 2, 3, 4) # r, c, t, h, w
        return frames


class TransposeReshape_2Bcthw:
    """
        bLhw or b,c,r,t,h,w -> B,c,t,h,w
         L:c*r*t               B:b*r
    """

    def __init__(self, mode, batch_size, repeat_aug, num_frames=16, c=3):
        self.mode = mode
        assert self.mode in ["pretrain", "finetune", "val", "test"]
        self.b = batch_size
        self.r = repeat_aug
        self.t = num_frames
        self.c = c
        self.l = repeat_aug * num_frames * c

    def __call__(self, frames, label=None, ids_keep=None, ids_restore=None, mask=None):  # bLhw
        if len(frames.shape) == 4:
            b, l, h, w = frames.shape
            frames = frames.reshape(self.b, self.r, self.c, self.t, h, w)  # b, c, r, t, h, w
        if len(frames.shape) == 6:
            b, r, c, t, h, w = frames.shape
            frames = frames.reshape(self.b * self.r, self.c, self.t, h, w)  # b*r, c, t, h, w

        if self.mode == "pretrain":
            assert ids_keep is not None \
                   and ids_restore is not None \
                   and mask is not None
            b, r, l = ids_keep.shape
            ids_keep = ids_keep.reshape(b * r, l)
            b, r, l = ids_restore.shape
            ids_restore = ids_restore.reshape(b * r, l)
            b, r, l = mask.shape
            mask = mask.reshape(b * r, l)
            return frames.astype(np.float32), ids_keep, ids_restore, mask
        else:
            assert label is not None
            if self.mode in ["val", "test"]:
                label = label.reshape(-1)
            return frames.astype(np.float32), label


class Crop:
    def __init__(self, size=224):
        self._size = size

    def __call__(self, frames):
        frames = frames[:self._size, :self._size, :]
        return frames


class TestAUG:
    def __init__(self,
                 rescale=1.0 / 255.0,
                 shift=0,
                 mean=(0.45, 0.45, 0.45),
                 std=(0.225, 0.225, 0.225),
                 repeat_aug=1,
                 short_side_size=224,
                 interpolation=Inter.BILINEAR,
                 cropsize=224,
                 test_num_spatial_crops=3,
                 num_frames=16):
        self._rescale = rescale
        self._shift = shift
        self._mean = mean
        self._std = std
        self._repeat_aug = repeat_aug
        self._short_side_size = short_side_size
        self._interpolation = interpolation
        self._cropsize = cropsize
        self._test_num_spatial_crops = test_num_spatial_crops
        self._num_frames = num_frames

        # self.rescale = vision.Resize(rescale=self._rescale, shift=self._shift)
        # self.normalize = vision.Normalize(mean=self._mean, std=self._std)
        # self.TransposeReshape = TransposeReshape(self._repeat_aug)
        # self.resize = vision.Resize(size=self._short_side_size, interpolation=self._interpolation)
        self.aug1 = Compose([vision.Rescale(rescale=self._rescale, shift=self._shift),
                             NumpyNormalize(mean=self._mean, std=self._std),
                             TransposeReshape(self._repeat_aug),
                             vision.Resize(size=self._short_side_size, interpolation=self._interpolation)])
        self.aug2 = Compose([vision.HWC2CHW(),
                             ReshapeTranspose(repeat_aug=self._repeat_aug, num_frames=self._num_frames)])

    def __call__(self, frames, labels, sample, index):
        frames = self.aug1(frames)
        h, w, c = frames.shape
        test_num_spatial_crops = self._test_num_spatial_crops
        # 如果只测试单个视角，则和验证集增强相同，采用中心裁剪，所以test_num_spatial_crops=3
        if test_num_spatial_crops == 1:
            test_num_spatial_crops = 3
            spatial_sample_index = 1
        else:
            spatial_sample_index = index % test_num_spatial_crops if test_num_spatial_crops > 1 else 1
        spatial_step = 1.0 * (max(h, w) - self._short_side_size) / (test_num_spatial_crops - 1)
        spatial_start = int(spatial_sample_index * spatial_step)
        if h >= w:
            frames = frames[spatial_start:spatial_start + self._short_side_size, :, :]
        else:
            frames = frames[:, spatial_start:spatial_start + self._short_side_size, :]
        frames = self.aug2(frames)
        return frames, labels, sample, index


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips, use_offset=False):
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips == 1:
                # Take the center clip if num_clips is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return int(start_idx), int(end_idx)


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    index = np.linspace(start_idx, end_idx, num_samples).astype(int)
    index = np.clip(index, 0, frames.shape[0] - 1).astype(np.long)
    new_frames = frames[index, :, :, :]
    return new_frames


def temporal_sampling_gray(frames, start_idx, end_idx, num_samples):
    index = np.linspace(start_idx, end_idx, num_samples).astype(int)
    index = np.clip(index, 0, frames.shape[0] - 1).astype(np.long)
    new_frames = frames[index, :, :]
    return new_frames


def rescale(frames, rescale_size=1.0 / 255.0, shift=0):
    rescale = vision.Rescale(rescale_size, shift)
    frames = rescale(frames)
    return frames


def normalize(frames, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    normalize = vision.Normalize(mean=mean, std=std)
    frames = normalize(frames)
    return frames


class NumpyNormalize:
    def __init__(self, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]):
        if isinstance(mean, list) or isinstance(mean, tuple):
            mean = np.array(mean)
        if isinstance(std, list) or isinstance(std, tuple):
            std = np.array(std)
        self._mean = mean
        self._std = std

    def __call__(self, frames):
        frames = frames - self._mean
        frames = frames / self._std
        return frames


def spatial_sampling(frames, spatial_idx=-1, scale=(0.5, 1.0), ratio=(0.75, 1.333), crop_size=224):
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames = random_resized_crop(frames, scale, ratio, crop_size)
    return frames


def random_resized_crop(frames, scale=(0.5, 1.0), ratio=(0.75, 1.333), crop_size=224):
    T, H, W, C = frames.shape
    start_x, start_y, target_h, target_w = _get_param_spatial_crop(scale, ratio, H, W)
    crop = vision.Crop((start_x, start_y), (target_h, target_w))
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    resize = vision.Resize(crop_size)

    new_frames = []
    for i in range(T):
        new_frames.append(resize(crop(frames[i])))
    new_frames = np.array(new_frames)
    return new_frames


def random_horizontal_flip(frames, prob=0.5):
    T, _, _, _ = frames.shape
    flip = vision.HorizontalFlip()
    if np.random.uniform() < prob:
        for i in range(T):
            frames[i] = flip(frames[i])
    return frames


def _get_param_spatial_crop(scale, ratio, height, width, num_repeat=10, log_scale=True):
    """
    Given scale, ratio, height and width, return sampled coordinates of the videos.
    """
    for _ in range(num_repeat):
        area = height * width
        target_area = random.uniform(*scale) * area
        if log_scale:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


class IdentityOperator:
    @staticmethod
    def __call__(input_tensor):
        return input_tensor

