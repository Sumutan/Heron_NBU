import numpy as np
from src.models_mae_v2_view import PretrainVisionTransformer
from src.video_vit import PatchEmbed

class patchifyTest():
    def patchify(self, imgs):
        """
        imgs: (N, 3, T, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        petch_info = N, _, T, H, W
        p = 2
        u = 2
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(N, 3, t, u, h, p, w, p)
        x = x.transpose(0, 2, 4, 6, 3, 5, 7, 1)  # (N, t, h, w, u, p, p, 3)
        x = x.reshape(N, t * h * w, u * p ** 2 * 3)
        return x, petch_info


    def unpatchify(self, x, patch_info):
        """
        x: (N, L, patch_size**2 *3)
        output_shape: (N, 3, T, H, W)
        """
        N, _, T, H, W = patch_info  # (N, 3, T, H, W)
        p = 2
        u = 2
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = x.reshape(N, t, h, w, u, p, p, 3)
        x = x.transpose(0, 7, 1, 4, 2, 5, 3, 6)  # (N, 3, t, u, h, p, w, p)
        x = x.reshape(patch_info)
        return x

if __name__ == '__main__':
    # 生成一个形状为 (2, 3, 2, 4, 4) 的自然数数组
    arr = np.arange(2 * 3 * 2 * 4 * 4).reshape((2, 3, 2, 4, 4))

    patchifier=patchifyTest()
    x,petch_info=patchifier.patchify(arr)

    x = patchifier.unpatchify(x,petch_info)

    print("done")