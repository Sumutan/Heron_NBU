import mindspore as ms
from src import models_mae  # 可以直接将src文件临时复制到同级文件夹
from src import models_vit

import torch

ckpt = 'k400finetune_B.ckpt'
params = ms.load_checkpoint(ckpt)

model = models_vit.__dict__['mae_vit_base_patch16']()

msg = ms.load_param_into_net(model, params)

print(msg)
