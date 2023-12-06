"""
掩码策略测试脚本
"""
import mindspore as ms
import numpy as np
import matplotlib.pyplot as plt


class MaskTest:
    def __init__(self, mask_ratio=0.9, hwc=(3, 3, 2), n=1):
        self._mask_ratio = mask_ratio

        self.mask = None
        self.ids_restore = None
        self.ids_keep = None
        self.ids_mask = None

        self.h = hwc[0]
        self.w = hwc[1]
        self.t = hwc[2]

        self.L = self.h * self.w * self.t
        self.N = n

    def show(self):
        # print("x_masked:", self.x_masked)
        print("mask:", self.mask)
        print("ids_restore:", self.ids_restore)
        print("ids_keep:", self.ids_keep)
        print("ids_mask:", self.ids_mask)
        print(self.mask.shape, self.ids_restore.shape, self.ids_keep.shape, self.ids_mask.shape)

    def draw_mask(self):
        # h = math.sqrt(self.mask.shape[-1])
        h, t = self.h, self.t
        assert h * h * t == self.mask.shape[-1], "h * h * t  != L"

        if len(self.mask.shape) > 1:
            mask_img = self.mask[0].reshape(t, h, h)
        else:
            mask_img = self.mask.reshape(t, h, h)

        # draw
        for i in range(mask_img.shape[0]):
            plt.imshow(mask_img[i, :, :], cmap='gray')
            plt.show()

    def random_masking(self, N, L):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        len_keep = int(L * (1 - self._mask_ratio))
        # if len_keep % 2 == 0:
        #     len_keep += 1

        noise = np.random.rand(N, L)  # np.random.rand(L)
        # noise = np.expand_dims(np.arange(0, L), axis=0).repeat(N, axis=0) # 前向对齐用

        # sort noise for each sample——
        ids_shuffle = np.argsort(noise, axis=-1).astype(np.int32)
        ids_restore = np.argsort(ids_shuffle, axis=-1).astype(np.int32)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]

        mask = np.ones((N, L), dtype=np.float32)
        mask[:, :len_keep] = 0.0

        for i in range(N):
            mask[i] = mask[i, ids_restore[i]]

        self.mask = mask
        self.ids_restore = ids_restore
        self.ids_keep = ids_keep
        self.ids_mask = ids_mask

        return mask, ids_restore, ids_keep, ids_mask

    def tube_masking(self, N, L):
        """
        Perform per-sample tube masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        num_patches_per_frame = self.w ** 2
        L_per_frame=num_patches_per_frame
        len_keep_per_frame = int(L_per_frame * (1 - self._mask_ratio))
        len_keep=len_keep_per_frame*self.t

        noise = np.random.rand(N, L_per_frame)
        noise =np.tile(noise,(1,self.t))

        ids_shuffle = np.argsort(noise, axis=-1).astype(np.int32)
        ids_restore = np.argsort(ids_shuffle, axis=-1).astype(np.int32)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]

        mask = np.ones((N, L), dtype=np.float32)
        mask[:, :len_keep] = 0.0

        for i in range(N):
            mask[i] = mask[i, ids_restore[i]]

        self.mask = mask
        self.ids_restore = ids_restore
        self.ids_keep = ids_keep
        self.ids_mask = ids_mask

        return mask, ids_restore, ids_keep, ids_mask

    def random_masking_lastframe(self,x, masktoken=1 * 14 * 14):  # mask最后1帧
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = L - masktoken
        assert len_keep >= 0
        print(len_keep)

        noise = ms.Tensor(np.random.rand(N, L), ms.float32)
        print(noise)

        # sort noise for each sample
        ids_shuffle = ms.Tensor(np.arange(N * L).reshape(N, L), ms.float32)
        # print(ids_shuffle)
        ids_restore = ids_shuffle
        # print(ids_restore)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep_1 = ids_keep.expand_dims(-1)  # ids_keep:(N,len_keep)->(N,len_keep,1)  <Tensor>
        ids_keep_2 = ids_keep_1.repeat(D, axis=-1).astype(ms.int32)
        x_masked = x.gather_elements(dim=1, index=ids_keep_2)

        mask = ms.Tensor(np.ones((N, L)), ms.float32)
        mask[:, :len_keep] = 0
        mask = mask.gather_elements(dim=1, index=ids_restore.astype(ms.int32))

        self.mask = mask
        self.ids_restore = ids_restore
        self.ids_keep = ids_keep
        # self.ids_mask = ids_mask

        return x_masked, mask, ids_restore, ids_keep




if __name__ == '__main__':
    # 1, 5, 5
    # x = ms.Tensor(np.array([[[0, 0, 0, 0, 0],
    #                          [1, 1, 1, 1, 1],
    #                          [2, 2, 2, 2, 2],
    #                          [3, 3, 3, 3, 3],
    #                          [4, 4, 4, 4, 4]]]), dtype=ms.float32)
    #
    # # x_masked, mask, ids_restore, ids_keep=random_masking(x, 0.5)
    # x_masked, mask, ids_restore, ids_keep = random_masking(x, 2)

    mask = MaskTest(mask_ratio=0.9,hwc=(14,14,8))
    mask.random_masking(1, mask.L)
    # mask.tube_masking(1, mask.L)
    # mask.random_masking_lastframe()

    def set_to_one(my_list):
        for i in range(len(my_list)):
            if isinstance(my_list[i], list):
                set_to_one(my_list[i])  # 递归处理嵌套列表
            else:
                my_list[i] = 255  # 将非列表元素替换为1
    set_to_one(mask.mask.reshape(8,14,14))
    mask.show()
    mask.draw_mask()
