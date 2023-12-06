from functools import partial

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import TruncatedNormal, Zero, initializer

import src.video_vit as video_vit


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


class PretrainVisionTransformerEncoder(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False, num_frames=16, dtype=ms.float16, cls_embed=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = video_vit.PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, t_patch_size=tubelet_size,
            frames=num_frames
        )
        num_patches = self.patch_embed.num_patches
        self.cls_embed = cls_embed
        if self.cls_embed:
            self.cls_token = init_params((1, 1, embed_dim), name='cls_token')
            num_patches += 1
        self.dtype = dtype
        if use_learnable_pos_emb:
            self.pos_embed = init_params((1, num_patches + 1, embed_dim), initializer_type=Zero(), dtype=self.dtype)
        else:
            # sine-cosine positional embeddings  加上了cls token
            self.pos_embed = video_vit.get_sinusoid_encoding_table(num_patches, embed_dim, dtype=self.dtype)

        self.linspace = ops.LinSpace()
        dpr = [x for x in ops.LinSpace()(ms.Tensor(0.), ms.Tensor(drop_path_rate), depth).asnumpy()]
        self.blocks = nn.CellList([
            video_vit.TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, attn_func=video_vit.Attention_v2, dtype=self.dtype
            )
            for i in range(depth)])
        self.norm = norm_layer((embed_dim,)).to_float(ms.float32)
        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.cat = ops.Concat(axis=1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, ids_keep):
        x = self.patch_embed(x)

        N, T, L, C = x.shape
        x = x.reshape(N, T * L, C)

        if self.cls_embed:
            x = x + self.pos_embed.astype(x.dtype)[:, 1:, :]
        else:
            x = x + self.pos_embed.astype(x.dtype)

        B, _, C = x.shape
        x_vis = x.gather_elements(dim=1, index=ids_keep.expand_dims(-1).repeat(C, axis=-1))

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_token = ops.cast(cls_token, x.dtype)
            cls_tokens = cls_token.broadcast_to((x.shape[0], -1, -1)) + self.pos_embed.astype(x.dtype)[:, 0, :]
            x_vis = self.cat((cls_tokens, x_vis))

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)

        # 现在无论是否有cls token都要把cls token输出
        return x_vis

    def construct(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, dtype=ms.float16, ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.dtype = dtype
        self.linspace = ops.LinSpace()
        dpr = [x for x in ops.LinSpace()(ms.Tensor(0.), ms.Tensor(drop_path_rate), depth).asnumpy()]
        self.blocks = nn.CellList([
            video_vit.TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, attn_func=video_vit.Attention_v2, dtype=self.dtype,
            )
            for i in range(depth)])
        self.norm = norm_layer((embed_dim,)).to_float(ms.float32)
        self.head = nn.Dense(embed_dim, num_classes).to_float(ms.float16) if num_classes > 0 else nn.Identity()

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def construct(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class PretrainVisionTransformer(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=1536,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 pred_t_dim=8,
                 norm_pix_loss=True,
                 cls_embed=False,
                 **kwargs):
        super().__init__()
        self.num_frames = 16
        self.dtype = ms.float16
        self.cls_embed = cls_embed
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            num_frames=self.num_frames,
            dtype=self.dtype,
            cls_embed=cls_embed,
        )
        self.t_pred_patch_size = tubelet_size * pred_t_dim // self.encoder.patch_embed.frames
        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            dtype=self.dtype,
        )
        self.encoder_to_decoder = nn.Dense(encoder_embed_dim, decoder_embed_dim, has_bias=False).to_float(ms.float16)

        self.mask_token = init_params((1, 1, decoder_embed_dim), initializer_type=Zero(), dtype=self.dtype)

        # 如果有cls_embed，pos_embed长度+1
        if cls_embed:
            self.pos_embed = video_vit.get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches + 1,
                                                                   decoder_embed_dim,
                                                                   dtype=self.dtype)
        else:
            self.pos_embed = video_vit.get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches,
                                                                   decoder_embed_dim,
                                                                   dtype=self.dtype)

        self.norm_pix_loss = norm_pix_loss

        self.mse_loss = nn.MSELoss()
        self.mean = ms.Tensor(kwargs.get('mean'))
        self.std = ms.Tensor(kwargs.get('std'))

        #op
        self.expand_dims = ops.ExpandDims()
    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward_loss(self, imgs, pred, ids_mask):
        N, C, T, H, W = imgs.shape
        p = self.encoder.patch_embed.patch_size[0]
        u = self.t_pred_patch_size
        h = w = H // p
        t = T // u
        target = imgs.reshape(N, C, t, u, h, p, w, p)
        target = target.transpose(0, 2, 4, 6, 3, 5, 7, 1)  # (N, t, h, w, u, p, p, 3)
        target = target.reshape(N, t * h * w, u * p * p * C)

        if self.norm_pix_loss:
            mean = target.mean(axis=-1, keep_dims=True)
            var = target.var(axis=-1, keepdims=True)
            target = (target - mean) / (var ** 0.5 + 1.0e-6)

        B, _, C = target.shape
        target = target.gather_elements(dim=1, index=ids_mask.expand_dims(-1).repeat(C, axis=-1)).view(B, -1, C)

        loss = self.mse_loss(pred, target)
        return loss

    def construct(self, x, mask, ids_restore, ids_keep, ids_mask):
        if len(x.shape) == 4:
            b, l, h, w = x.shape
            x = x.reshape(b, 3, self.num_frames, -1, h, w)
            x = x.transpose(0, 3, 1, 2, 4, 5)  # b, r, c, t, h, w
        if len(x.shape) == 6:
            b, r, c, t, h, w = x.shape
            x = x.reshape(b * r, c, t, h, w)
        if len(ids_keep.shape) == 3:
            b, r, l = ids_keep.shape
            ids_keep = ids_keep.reshape(b * r, l)
            b, r, l = ids_mask.shape
            ids_mask = ids_mask.reshape(b * r, l)
        imgs = x

        x_vis = self.encoder(x, ids_keep)  # [B, N_vis+(cls_token), C_e]

        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis+(cls_token), C_e]
        # 这里原来是给vis_token设计的特征映射用在cls_token上不确定是否有效

        B, N, C = x_vis.shape
        expand_pos_embed = self.pos_embed.broadcast_to((B, -1, -1))


        if self.cls_embed:
            pos_emd_vis = expand_pos_embed[:, 1:, :] \
                .gather_elements(dim=1, index=ids_keep.expand_dims(-1).repeat(C, axis=-1)) \
                .view(B, -1, C)
            pos_emd_mask = expand_pos_embed[:, 1:, :] \
                .gather_elements(dim=1, index=ids_mask.expand_dims(-1).repeat(C, axis=-1)) \
                .view(B, -1, C)
            x_vis_addposition = ops.concat(
                        (self.expand_dims(x_vis[:, 0, :] + expand_pos_embed[:, 0, :],1), x_vis[:, 1:, :] + pos_emd_vis)
                        ,axis=1)
            x_full = ops.concat((x_vis_addposition, self.mask_token + pos_emd_mask), axis=1)  # [B, N, C_d]
        else:
            pos_emd_vis = expand_pos_embed.\
                gather_elements(dim=1,index=ids_keep.expand_dims(-1).repeat(C, axis=-1)).view(B,-1,C)
            pos_emd_mask = expand_pos_embed.\
                gather_elements(dim=1,index=ids_mask.expand_dims(-1).repeat(C, axis=-1)).view(B,-1,C)
            x_full = ops.Concat(axis=1)((x_vis + pos_emd_vis, self.mask_token + pos_emd_mask))  # [B, N, C_d]

        x = self.decoder(x_full, pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16] w/o cls token

        loss = self.forward_loss(imgs, x, ids_mask)
        return loss
