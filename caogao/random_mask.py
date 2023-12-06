import mindspore as ms
from mindspore import ops
import numpy as np
from pprint import pp


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = L * (1 - mask_ratio)
    len_keep = int(len_keep)
    print(len_keep)

    noise = ms.Tensor(np.random.rand(N, L), ms.float32)
    print(noise)

    # sort noise for each sample
    sort = ops.Sort()
    ids_shuffle = sort(noise)[1]
    print(ids_shuffle)
    ids_restore = sort(ms.Tensor(ids_shuffle, ms.float32))[1]
    print(ids_restore)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep_1 = ids_keep.expand_dims(-1)
    ids_keep_2 = ids_keep_1.repeat(D, axis=-1).astype(ms.int32)
    x_masked = x.gather_elements(dim=1, index=ids_keep_2)

    mask = ms.Tensor(np.ones((N, L)), ms.float32)
    mask[:, :len_keep] = 0
    mask = mask.gather_elements(dim=1, index=ids_restore.astype(ms.int32))

    return x_masked, mask, ids_restore, ids_keep


# mask最后1帧
def random_masking_lastframe(x, masktoken=1 * 14 * 14):
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
    print(ids_shuffle)
    ids_restore = ids_shuffle
    print(ids_restore)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep_1 = ids_keep.expand_dims(-1)  # ids_keep:(N,len_keep)->(N,len_keep,1)  <Tensor>
    ids_keep_2 = ids_keep_1.repeat(D, axis=-1).astype(ms.int32)
    x_masked = x.gather_elements(dim=1, index=ids_keep_2)

    mask = ms.Tensor(np.ones((N, L)), ms.float32)
    mask[:, :len_keep] = 0
    mask = mask.gather_elements(dim=1, index=ids_restore.astype(ms.int32))

    return x_masked, mask, ids_restore, ids_keep


if __name__ == '__main__':
    # 1, 5, 5
    x = ms.Tensor(np.array([[[0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1],
                             [2, 2, 2, 2, 2],
                             [3, 3, 3, 3, 3],
                             [4, 4, 4, 4, 4]]]), dtype=ms.float32)

    # x_masked, mask, ids_restore, ids_keep=random_masking(x, 0.5)
    x_masked, mask, ids_restore, ids_keep = random_masking_lastframe(x, 2)

    print("x_masked:", x_masked)
    print("mask:", mask)
    print("ids_restore:", ids_restore)
    print("ids_keep:", ids_keep)
    print(x_masked.shape, mask.shape, ids_restore.shape, ids_keep.shape)

    # input_length = 2 * 16 * 14 * 14 * 512  # 指定输入长度
    # tensor_shape = (2, 16 * 14 * 14, 512)  # 指定张量形状 x: [N, L, D]
    #
    # array = np.random.rand(input_length)  # 生成指定长度的一维数组
    # tensor = np.reshape(array, tensor_shape)  # 将一维数组reshape成指定形状的张量
    #
    # # 输出张量
    # pp(tensor)
    # pp(tensor.shape)
