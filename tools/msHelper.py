"""
Build tools for MS
Compensating for the lack of ease of use in the MindSpore.
"""
import mindspore as ms
import numpy as np


def randn(*size, dtype=ms.float32):
    """
    生成一个指定形状的张量，其元素从标准正态分布（均值为0，标准差为1）中随机抽取。

    :param size: 可变长度的整数序列，指定生成张量的维度。例如，randn(2, 3) 会生成一个 2x3 的张量。
    :param dtype: 生成的张量的数据类型，默认为 mindspore.float32。可以指定为其他数据类型，如 mindspore.float16。
    :return: 符合标准正态分布的随机张量，具有指定的形状和数据类型。
    """
    tensor = ms.Tensor(np.random.randn(*size), dtype=dtype)
    return tensor


if __name__ == '__main__':
    randn(3,4,5)