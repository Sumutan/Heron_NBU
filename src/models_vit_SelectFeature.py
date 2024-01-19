"""
特征选择版本的vit
与原先对于所有输出token做平均池化不同（B,N,C）->(N,1,C)
修改N个特征的融合方式以替代平均池化
"""

import time
from functools import partial

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Parameter, Tensor, context
from mindspore.common.initializer import (Normal, One, TruncatedNormal,
                                          XavierUniform, Zero, initializer)

import src.video_vit as video_vit

# context.set_context(mode=context.GRAPH_MODE)


class Feature_selector(nn.Cell):
    def __init__(self,
                 input_dim=768):
        pass

    def construct(self,x):  # x:[B,N,C]
        '''
        :param x: inout feature [B,N,C]
        :return: selected feature [B,1,C]
        '''


class VisionTransformer(nn.Cell):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            num_frames=16,
            t_patch_size=4,  # patch的时间维数
            no_qkv_bias=False,
            qk_scale=None,
            trunc_init=False,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            fc_drop_rate=0.5,
            sep_pos_embed=True,
            cls_embed=True,
            num_classes=101,
            freeze_encoder=False,
            use_mean_pooling=True,
            use_learnable_pos_emb=False,
            init_scale=0.0,
            **kwargs
    ):
        super(VisionTransformer, self).__init__()

        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        # op
        self.cast = ops.Cast()
        self.cat = ops.Concat(axis=1)
        self.linspace = ops.LinSpace()
        self.mul = ops.Mul()

        self.dtype = ms.float16
        if norm_layer == 'LayerNorm':
            norm_layer = nn.LayerNorm

        self.patch_embed = video_vit.PatchEmbed(img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size)
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        # Encoder Params
        if self.cls_embed:
            # self.cls_token = self.init_params((1, 1, embed_dim),dtype=self.dtype)
            self.cls_token = self.init_params((1, 1, embed_dim),name='cls_token')
        if self.sep_pos_embed:
            self.pos_embed_spatial = self.init_params((1, input_size[1] * input_size[2], embed_dim))
            self.pos_embed_temporal = self.init_params((1, input_size[0], embed_dim))
            if self.cls_embed:
                self.pos_embed_class = self.init_params((1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            if use_learnable_pos_emb:
                self.pos_embed = self.init_params((1, _num_patches, embed_dim))
            else:
                self.pos_embed = video_vit.get_sinusoid_encoding_table(_num_patches, embed_dim)

        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        drop_path_rate = Tensor(drop_path_rate, ms.float32)
        dpr = [float(x) for x in
               self.linspace(Tensor(0, ms.float32), drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.CellList(
            [
                video_vit.TransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    dtype=self.dtype
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer((embed_dim,)).to_float(ms.float32)
        self.fc_norm = norm_layer((embed_dim,)).to_float(ms.float32) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(keep_prob=1.0 - fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Dense(embed_dim, num_classes).to_float(self.dtype)

        self.init_weight()
        self.head.weight.set_data(self.mul(self.head.weight.data, init_scale))
        self.head.bias.set_data(self.mul(self.head.bias.data, init_scale))
        # self.loss = nn.CrossEntropyLoss()
        if freeze_encoder:
            self.freeze_encoder()
        print("model initialized")

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

    def init_weight(self):
        # init weight
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                if self.trunc_init:
                    cell.weight.set_data(initializer(TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
                else:
                    cell.weight.set_data(initializer(XavierUniform(), cell.weight.shape, cell.weight.dtype))

                if cell.bias is not None:
                    cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer(One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer(Zero(), cell.beta.shape, cell.beta.dtype))
            # elif isinstance(cell, nn.Conv3d):
            #     if self.trunc_init:
            #         cell.weight.set_data(initializer(TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
            #     else:
            #         cell.weight.set_data(initializer(XavierUniform(), cell.weight.shape, cell.weight.dtype))

    def freeze_encoder(self):
        for name, param in self.parameters_and_names():
            if 'head' not in name:
                param.requires_grad = False

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape
        x = x.reshape(N, T * L, C)

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_token = self.cast(cls_token, self.dtype)  # add
            cls_tokens = cls_token.broadcast_to((x.shape[0], -1, -1))
            x = self.cat((cls_tokens, x))

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(self.input_size[0], axis=0).reshape(1, -1, self.embed_dim) \
                        + self.pos_embed_temporal.repeat(self.input_size[1] * self.input_size[2], axis=1)
            pos_embed = pos_embed.broadcast_to((x.shape[0], -1, -1))
            if self.cls_embed:
                pos_embed = self.cat((self.pos_embed_class.broadcast_to((pos_embed.shape[0], -1, -1)), pos_embed))
        else:
            pos_embed = self.pos_embed[:, :, :]

        x = x + pos_embed
        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def construct(self, imgs, labels):
        imgs = imgs.astype(ms.float32)
        if len(imgs.shape) == 6:
            b, r, c, t, h, w = imgs.shape
            imgs = imgs.reshape(b * r, c, t, h, w)
            if labels.shape[1] != self.num_classes:
                labels = labels.reshape(b * r)

        x = self.forward_encoder(imgs)
        x = self.cast(x, self.dtype)
        logits = self.head(self.fc_dropout(x))
        logits = self.cast(logits, ms.float32)
        # loss = self.loss(logits, labels)

        return logits, labels


class VisionTransformer_v2(nn.Cell):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            num_frames=16,
            t_patch_size=4,  # patch的时间维数
            no_qkv_bias=False,
            qk_scale=None,
            trunc_init=False,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            fc_drop_rate=0.5,
            sep_pos_embed=True,
            cls_embed=False,
            num_classes=400,
            freeze_encoder=False,
            use_mean_pooling=True,
            use_learnable_pos_emb=False,
            init_scale=0.0,
            **kwargs
    ):
        super(VisionTransformer_v2, self).__init__()

        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        # op
        self.cast = ops.Cast()
        self.cat = ops.Concat(axis=1)
        self.linspace = ops.LinSpace()
        self.mul = ops.Mul()

        self.dtype = ms.float16
        if norm_layer == 'LayerNorm':
            norm_layer = nn.LayerNorm

        self.patch_embed = video_vit.PatchEmbed(img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size)
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        assert cls_embed ^ use_mean_pooling  # 或非逻辑：use_class_token和use_mean_pooling必须开其中一个

        self.cls_token=None
        if cls_embed:
            self.cls_token = self.init_params((1, embed_dim), name='cls_token')  # 此处sigma还不确定

        if use_learnable_pos_emb:
            self.pos_embed = self.init_params((1, num_patches, embed_dim))
        else:
            if cls_embed:
                self.pos_embed = video_vit.get_sinusoid_encoding_table(num_patches + 1, embed_dim)
            else:
                self.pos_embed = video_vit.get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        drop_path_rate = Tensor(drop_path_rate, ms.float32)
        dpr = [float(x) for x in
               self.linspace(Tensor(0, ms.float32), drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.CellList(
            [
                video_vit.TransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attn_func=video_vit.Attention_v2,
                    dtype=self.dtype
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer((embed_dim,)).to_float(ms.float32)
        self.fc_norm = norm_layer((embed_dim,)).to_float(ms.float32) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(keep_prob=1.0 - fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Dense(embed_dim, num_classes).to_float(self.dtype)

        self.init_weight()
        self.head.weight.set_data(self.mul(self.head.weight.data, init_scale))
        self.head.bias.set_data(self.mul(self.head.bias.data, init_scale))
        # self.loss = nn.CrossEntropyLoss()
        if freeze_encoder:
            self.freeze_encoder()
        print("model initialized")

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

    def init_weight(self):
        # init weight
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                if self.trunc_init:
                    cell.weight.set_data(initializer(TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
                else:
                    cell.weight.set_data(initializer(XavierUniform(), cell.weight.shape, cell.weight.dtype))

                if cell.bias is not None:
                    cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer(One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer(Zero(), cell.beta.shape, cell.beta.dtype))
            # elif isinstance(cell, nn.Conv3d):
            #     if self.trunc_init:
            #         cell.weight.set_data(initializer(TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
            #     else:
            #         cell.weight.set_data(initializer(XavierUniform(), cell.weight.shape, cell.weight.dtype))

    def freeze_encoder(self):
        for name, param in self.parameters_and_names():
            if 'head' not in name:
                param.requires_grad = False

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape
        x = x.reshape(N, T * L, C)

        if self.cls_token is not None:
            cls_tokens = ops.tile(self.cls_token.astype(x.dtype), (x.shape[0], 1, 1))
            x = ops.concat((cls_tokens, x), axis=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed.broadcast_to((x.shape[0], -1, -1))

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.cls_token is None:  # self.fc_norm is not None mean use_mean_pooling=True


            x_selected=x[:, 0]
            return x_selected
            # return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def construct(self, imgs, labels):
        imgs = imgs.astype(ms.float32)
        if len(imgs.shape) == 4:
            b, l, h, w = imgs.shape
            imgs = imgs.reshape(b, 3, self.num_frames, -1, h, w)
            imgs = imgs.transpose(0, 3, 1, 2, 4, 5)  # b, r, c, t, h, w
        if len(imgs.shape) == 6:
            b, r, c, t, h, w = imgs.shape
            imgs = imgs.reshape(b * r, c, t, h, w)
            if labels.shape[1] != self.num_classes:
                labels = labels.reshape(b * r)

        x = self.forward_encoder(imgs)
        x = self.cast(x, self.dtype)
        logits = self.head(self.fc_dropout(x))
        logits = self.cast(logits, ms.float32)
        # loss = self.loss(logits, labels)

        return logits, labels


def mae_vit_base_patch16():
    model = VisionTransformer_v2(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
    )
    return model

