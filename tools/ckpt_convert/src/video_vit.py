import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import Parameter, Tensor, context, ops
from mindspore.common.initializer import (Normal, One, TruncatedNormal,
                                          XavierUniform, Zero, initializer)
from mindspore.ops import functional as F
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE)


class PatchEmbed(nn.Cell):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        # temporal related:
        frames=32,
        t_patch_size=2
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        assert frames % t_patch_size == 0

        print(f"img_size {img_size} patch_size {patch_size} frames {frames} t_patch_size {t_patch_size}")

        num_patches = ((img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (frames // t_patch_size))

        self.input_size = (frames // t_patch_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.img_size = img_size
        self.patch_size = patch_size

        self.embed_dim = embed_dim

        self.frames = frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]
        self.t_grid_size = frames // t_patch_size

        kernel_size = tuple([t_patch_size] + list(patch_size))
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size, has_bias=True).to_float(ms.float16)

    def construct(self, x):
        B, C, T, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], "模型参数[img_size]与输入视频大小[H,W]不相等"
        assert T == self.frames, "模型参数[frames]与输入视频帧数[T]不相等"

        x = self.proj(x)
        x = x.reshape(B, self.embed_dim, self.input_size[0], self.input_size[1] * self.input_size[2])
        x = x.transpose(0, 2, 3, 1)
        return x

def get_sinusoid_encoding_table(n_position, d_hid, dtype=ms.float32): 
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  ops.ExpandDims()(Parameter(Tensor(sinusoid_table, dtype=dtype), requires_grad=False), 0)

class Dropout(nn.transformer.layers._Dropout):
    # pylint: disable=W0212
    """
        A Dropout Implements with P.DropoutGenMask and  P.DropoutDoMask for context training.
    """
    def __init__(self, keep_prob=0.5, dtype=ms.float32):
        super(Dropout, self).__init__(keep_prob=keep_prob, dtype=dtype)

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob):
        super(DropPath, self).__init__()
        self.drop = Dropout(keep_prob=1 - drop_prob)
        self.mask = Tensor(np.ones(1,), dtype=ms.float32)
        self.tile = P.Tile()
        self.mul = P.Mul()

    def construct(self, x):
        if not self.training:
            return x
        mask = self.tile(self.mask, (x.shape[0],) + (1,) * (x.ndim-1))
        out = self.drop(mask)
        out = self.mul(out, x)
        return out

class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        dtype=ms.float32
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dtype = dtype

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias).to_float(dtype)
        self.k = nn.Dense(dim, dim, has_bias=qkv_bias).to_float(dtype)
        self.v = nn.Dense(dim, dim, has_bias=qkv_bias).to_float(dtype)

        self.softmax = nn.Softmax(axis=-1).to_float(ms.float32)

        self.attn_drop = nn.Dropout(keep_prob=1.0-attn_drop)  # do not use
        self.proj = nn.Dense(dim, dim).to_float(dtype)
        self.proj_drop = nn.Dropout(keep_prob=1.0-proj_drop)

        # op
        self.batchMatMul_trans_b = ops.BatchMatMul(transpose_b=True)
        self.batchMatMul = ops.BatchMatMul()

        self.cast = ops.Cast()

    def construct(self, x, B, N, C):
        # B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)

        attn = self.batchMatMul_trans_b(q, k) * self.scale

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = self.cast(attn, ms.float16)

        x = self.batchMatMul(attn, v).transpose((0, 2, 1, 3)).reshape((B, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B*N, C)
        return x

class Attention_v2(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        dtype=ms.float32
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dtype = dtype

        self.scale = qk_scale or head_dim ** -0.5

        self.cast = ops.Cast()
        self.concat_op = ops.Concat()
        self.qkv_bias = qkv_bias

        self.qkv = nn.Dense(dim, head_dim * self.num_heads * 3, has_bias=True).to_float(self.dtype)
        if qkv_bias:
            initial = initializer(Zero(), head_dim * self.num_heads, dtype=self.dtype)
            self.q_bias = Parameter(initial, requires_grad=True)
            self.k_bias = Parameter(initial, requires_grad=False)
            self.v_bias = Parameter(initial, requires_grad=True)
            q_k_v_bias = self.concat_op((self.q_bias, self.k_bias, self.v_bias))
            self.qkv.bias.set_data(q_k_v_bias)
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.softmax = nn.Softmax(axis=-1).to_float(ms.float32)

        self.attn_drop = nn.Dropout(keep_prob=1.0-attn_drop)  # do not use
        self.proj = nn.Dense(dim, dim).to_float(dtype)
        self.proj_drop = nn.Dropout(keep_prob=1.0-proj_drop)

        # op
        self.batchMatMul_trans_b = ops.BatchMatMul(transpose_b=True)
        self.batchMatMul = ops.BatchMatMul()

    def construct(self, x, B, N, C):
        # 下面的if在graph mode要注释掉，pynative mode要取消注释
        # if self.q_bias is not None:
        #     qkv_bias = self.concat_op((self.q_bias, self.k_bias, self.v_bias))
        #     self.qkv.bias.set_data(qkv_bias)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = self.batchMatMul_trans_b(q, k)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = self.cast(attn, ms.float16)

        x = self.batchMatMul(attn, v).transpose((0, 2, 1, 3)).reshape((B, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B*N, C)
        return x

class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, dtype=ms.float32):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Dense(in_features, hidden_features).to_float(dtype)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features).to_float(dtype)
        self.drop = nn.Dropout(1.0-drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
        init_values=0,
        dtype=ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.norm1 = norm_layer((dim,)).to_float(ms.float32)
        self.attn = attn_func(dim, 
                              num_heads=num_heads, 
                              qkv_bias=qkv_bias, 
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              dtype=self.dtype)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer((dim,)).to_float(ms.float32)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            dtype=self.dtype
        )
        if init_values > 0:
            self.gamma_1 = init_values * self.init_params((dim,), initializer_type=One())
            self.gamma_2 = init_values * self.init_params((dim,), initializer_type=One())
        else:
            self.gamma_1, self.gamma_2 = None, None

    @staticmethod
    def init_params(
            shape,
            name=None,
            initializer_type=TruncatedNormal(sigma=0.02),
            dtype=ms.float32,
            requires_grad=True
    ):
        initial = initializer(initializer_type, shape, dtype)
        initial.init_data()
        return ms.Parameter(initial, name=name, requires_grad=requires_grad)

    def construct(self, x):
        B, N, C = x.shape
        x = x.view(B*N, C)
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(F.cast(self.norm1(x), self.dtype), B, N, C))
            x = x + self.drop_path(self.mlp(F.cast(self.norm2(x), self.dtype)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(F.cast(self.norm1(x), self.dtype), B, N, C))
            x = x + self.drop_path(self.gamma_2 * self.mlp(F.cast(self.norm2(x), self.dtype)))
        x = x.view(B, N, C)
        return x

