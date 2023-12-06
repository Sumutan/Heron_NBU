"""
与models_vit不同于该model输出为feature，用于提取特征
idea测试：
    上一阶段使用 AISO（All Input and Some Output）取得阶段性效果后，证明在监控领域前景部分对应token具有高价值信息，只取这一部分的feature
    更加有效。
    本阶段尝试在 此基础上，通过在最后的用加权替代ViT传统的MeanPooling
全部特征->vit->取部分动态特征取Weighted Add，作为特征输出
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, context
from mindspore.common.initializer import (Normal, One, TruncatedNormal,
                                          XavierUniform, Zero, initializer)

import src.video_vit as video_vit

context.set_context(mode=context.GRAPH_MODE)


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
            cls_embed=True,
            num_classes=400,
            freeze_encoder=False,
            use_mean_pooling=True,
            use_learnable_pos_emb=False,
            init_scale=0.0,
            mask_ratio=0.0,
            **kwargs
    ):
        super(VisionTransformer_v2, self).__init__()

        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.mask_ratio = mask_ratio
        # op
        self.cast = ops.Cast()
        self.cat = ops.Concat(axis=1)
        self.linspace = ops.LinSpace()
        self.mul = ops.Mul()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.min = ops.ReduceMin(keep_dims=True)
        self.max = ops.ReduceMax(keep_dims=True)

        self.a = kwargs['alpha']  # alpha is structure weight scaling factor

        self.dtype = ms.float16
        if norm_layer == 'LayerNorm':
            norm_layer = nn.LayerNorm

        self.patch_embed = video_vit.PatchEmbed(img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size)
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if use_learnable_pos_emb:
            self.pos_embed = self.init_params((1, num_patches, embed_dim))
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
        self.norm_depth = norm_layer((int(1568 * (1 - self.mask_ratio)),)).to_float(ms.float32)
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

    def forward_encoder(self, x, ids_keep, depth):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape
        N_d, T_d, H_d, W_d = depth.shape
        x = x.reshape(N, T * L, C)
        x_depth = depth.reshape(N_d, T_d * H_d * W_d)  # .expand_as(x)
        x_depth = x_depth.gather_elements(dim=1, index=ids_keep)

        # MaxMin Norm
        # mean = self.mean(x_depth, 1)
        # max = self.max(x_depth, 1)
        # min = self.min(x_depth, 1)
        # x_depth = self.a * (x_depth - mean) / (max - min) + 1  # a:0.2/0.4/0.8
        # mean_ = self.mean(x_depth, 1)
        # max_ = self.max(x_depth, 1)
        # min_ = self.min(x_depth, 1)

        # Norm
        x_depth = self.a * self.norm_depth(x_depth) + 1  # a:0.1/0.2/0.05
        x_depth = x_depth.expand_dims(-1)

        if self.pos_embed is not None:
            x = x + self.pos_embed.broadcast_to((x.shape[0], -1, -1))
        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # 取得动态部分token作为输出
        B, _, C = x.shape
        x = x.gather_elements(dim=1, index=ids_keep.expand_dims(-1).repeat(C, axis=-1))
        x = self.mul(x, x_depth)

        x = self.norm(x)  # Identity
        # return x
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))  # Layer Norm
        else:
            return x[:, 0]

    def construct(self, imgs, ids_keep, depth):
        """
        input
            imgs:[B,c,t,h,w] 存放着图片的多维数组
            keep_ids: [B, L_keep]
            depth: [B,c,t,h,w] 存放着图片对应深度图的多维数组
        """
        imgs = imgs.astype(ms.float32)
        depth = depth.astype(ms.float32)

        x_vis = self.forward_encoder(imgs, ids_keep, depth)

        x_vis = self.cast(x_vis, self.dtype)
        feature_output = self.fc_dropout(x_vis)
        feature_output = self.cast(feature_output, ms.float32)

        return feature_output
