import time
from functools import partial

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import context
from mindspore.common.initializer import (TruncatedNormal, XavierUniform, Normal, Zero, One,
                                          initializer)

import src.video_vit as video_vit

context.set_context(mode=context.PYNATIVE_MODE)


class MaskedAutoencoderViT(nn.Cell):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            decoder_embed_dim=512,
            decoder_depth=4,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            norm_pix_loss=True,
            num_frames=16,
            t_patch_size=4,  # patch的时间维数
            patch_embed=video_vit.PatchEmbed,
            no_qkv_bias=False,
            trunc_init=False,
            sep_pos_embed=False,
            cls_embed=False,
            pred_t_dim=8,
            **kwargs
        ):
        super(MaskedAutoencoderViT, self).__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim
        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames


        if norm_layer == 'LayerNorm':
            norm_layer = nn.LayerNorm

        self.patch_embed = patch_embed(img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size)
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        # Encoder Params
        if self.cls_embed:
            self.cls_token = self.init_params((1, 1, embed_dim))

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
            
            self.pos_embed = self.init_params((1, _num_patches, embed_dim))

        self.blocks = nn.CellList(
            [
                video_vit.TransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer((embed_dim,))

        self.decoder_embed = nn.Dense(embed_dim, decoder_embed_dim, has_bias=True)

        if self.trunc_init:
            self.mask_token = self.init_params((1, 1, decoder_embed_dim))
        else:
            self.mask_token = self.init_params((1, 1, decoder_embed_dim), initializer_type=Normal(sigma=0.02))
            
        self.decoder_pos_embed = self.init_params((1, num_patches, decoder_embed_dim))

        self.decoder_blocks = nn.CellList(
            [
                video_vit.TransformerBlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        # Decoder Params

        self.decoder_norm = norm_layer((decoder_embed_dim,))
        self.decoder_pred = nn.Dense(decoder_embed_dim,
                                     self.t_pred_patch_size * patch_size ** 2 * in_chans,
                                     has_bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.init_weight()

        # op
        self.sort = ops.Sort()
        self.cat = ops.Concat(axis=1)
        self.repeat_elements = ops.repeat_elements
        self.linspace = ops.LinSpace()

        # other
        self._embed_dim = embed_dim
        self._decoder_embed_dim = decoder_embed_dim

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
            elif isinstance(cell, nn.Conv3d):
                if self.trunc_init:
                    cell.weight.set_data(initializer(TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
                else:
                    cell.weight.set_data(initializer(XavierUniform(), cell.weight.shape, cell.weight.dtype))

    def patchify(self, imgs):
        """
        imgs: (N, 3, T, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(N, 3, t, u, h, p, w, p)
        x = x.transpose(0, 2, 4, 6, 3, 5, 7, 1) # (N, t, h, w, u, p, p, 3)
        x = x.reshape(N, t * h * w, u * p ** 2 * 3)
        patch_info = (N, T, H, W, p, u, t, h, w)
        return x, patch_info
    
    def unpatchify(self, x, patch_info):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = patch_info

        x = x.reshape(N, t, h, w, u, p, p, 3)

        x = x.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        imgs = x.reshape(N, 3, T, H, W)
        return imgs

    def forward_encoder(self, x, ids_keep):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape
        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        x = x.gather_elements(dim=1, index=ids_keep.expand_dims(-1).repeat(C, axis=-1))

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.broadcast_to((x.shape[0], -1, -1))
            x = self.cat((cls_tokens, x))

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(self.input_size[0], axis=0).reshape(1, -1, self._embed_dim) \
                      + self.pos_embed_temporal.repeat(self.input_size[1] * self.input_size[2], axis=1)
            pos_embed = pos_embed.broadcast_to((x.shape[0], -1, -1))
            pos_embed = pos_embed.gather_elements(index=ids_keep.expand_dims(-1).repeat(pos_embed.shape[2], axis=-1), dim=1)
            if self.cls_embed:
                pos_embed = self.cat((self.pos_embed_class.broadcast_to((pos_embed.shape[0], -1, -1)), pos_embed))
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].broadcast_to((x.shape[0], -1, -1))
            pos_embed = pos_embed.gather_elements(index=ids_keep.expand_dims(-1).repeat(pos_embed.shape[2], axis=-1), dim=1)
            if self.cls_embed:
                pos_embed = self.cat((self.pos_embed[:, :1, :].broadcast_to((pos_embed.shape[0], -1, -1)), pos_embed))

        x = x.view(N, -1, C) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_decoder(self, x, ids_restore):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, axis=0).repeat(T * H * W + 0 - x.shape[1], axis=1) # 原来一共有T*H*W个patch
        x_ = self.cat((x[:, :, :], mask_tokens))  # no cls token
        x_ = x_.view(N, T * H * W, C)

        x_ = x_.gather_elements(dim=1, index=ids_restore.expand_dims(-1).repeat(x_.shape[2], axis=-1))
        x = x_.view(N, T * H * W, C)

        decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        return x

    def forward_loss(self, imgs, idx, pred, mask):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        _imgs = imgs[:, :, idx, :, :]

        target, patch_info = self.patchify(_imgs)
        var,mean=0.0,0.0
        if self.norm_pix_loss:
            mean = target.mean(axis=-1, keep_dims=True)
            var = target.var(axis=-1, keepdims=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(axis=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        if self.norm_pix_loss:
            pred = pred * (var + 1.0e-6) ** 0.5 + mean
        pred_video = self.unpatchify(pred, patch_info)

        return loss, pred_video

    def construct(self, imgs, mask, ids_restore, ids_keep, ids_pred):
        imgs = imgs.astype(ms.float32)
        if len(imgs.shape) == 6:
            b, r, c, t, h, w = imgs.shape
            imgs = imgs.reshape(b * r, c, t, h, w)
        if len(ids_keep.shape) == 3:
            b, r, l = ids_keep.shape
            ids_keep = ids_keep.reshape(b * r, l)
            b, r, l = ids_restore.shape
            ids_restore = ids_restore.reshape(b * r, l)
            b, r ,l = mask.shape
            mask = mask.reshape(b * r, l)
        if len(ids_pred.shape) == 2:
            ids_pred = ids_pred[0]

        latent = self.forward_encoder(imgs, ids_keep)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss, pred_video = self.forward_loss(imgs, ids_pred, pred, mask)

        raw_imgs = imgs[:, :, ids_pred, :, :]
        imgs, patch_info = self.patchify(raw_imgs)
        N, L, C = imgs.shape
        imgs = imgs.gather_elements(dim=1, index=ids_keep.expand_dims(-1).repeat(C, axis=-1))
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size
        masks = ms.Tensor(shape=(N, T * H * W - imgs.shape[1], C), dtype=ms.float32, init=Zero())
        imgs = self.cat((imgs[:, :, :], masks))  # no cls token
        imgs = imgs.gather_elements(dim=1, index=ids_restore.expand_dims(-1).repeat(imgs.shape[2], axis=-1))
        masked_imgs = self.unpatchify(imgs, patch_info)

        # pred_img = self.unpatchify(pred, patch_info)
        return loss, raw_imgs, masked_imgs, pred_video


def mae_vit_base_patch16():
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
    )
    return model


def mae_vit_large_patch16():
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
    )
    return model

def mae_vit_huge_patch14():
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
    )
    return model

def mae_vit_test():
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=1,
        num_heads=1,

        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=1,

        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
    )
    return model
