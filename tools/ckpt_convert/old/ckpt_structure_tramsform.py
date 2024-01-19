"""
将ckpt1中的权重转移到ckpt2中，ckpt1与ckpt2有些许区别
"""
import mindspore as ms
from src import models_mae  # 可以直接将src文件临时复制到同级文件夹
from src import models_vit

# import torch


def show_info(pth_file="mae_pretrain_vit_base.pth"):
    print("show_info:",pth_file)
    info_path = pth_file.split('.')[0] + '_info.txt'
    info_file = open(info_path, 'w')
    if '.ckpt' in pth_file:
        state_dict = ms.load_checkpoint(pth_file)
    # elif '.pth' in pth_file:
    #     state_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    elif 'module' in state_dict:
        state_dict = state_dict['module']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    for k, v in state_dict.items():
        info = k + ' ' + str(v.shape) + ' ' + str(v.dtype)
        info_file.write(info + '\n')
        # print(info)

ckpt1 = "model_ckpt_hkm.ckpt"
ckpt2 = "k400finetune_B.ckpt"

show_info(ckpt1)
show_info(ckpt2)

# print("ms.load_checkpoint")
# params1 = ms.load_checkpoint(ckpt1)
# params2 = ms.load_checkpoint(ckpt2)
#
# print("define model")
# print("model1")
# model1 = models_vit.__dict__['mae_vit_base_patch16']()
# print("model2")
# model2 = models_vit.__dict__['mae_vit_V2_base_patch16']()
#
# print("ms.load_param_into_net")
# msg1 = ms.load_param_into_net(model1, params1)      #4 parameters in the 'net' are not loaded,
# print(msg1)
# msg2 = ms.load_param_into_net(model2, params2)
# print(msg2)

#开始转移权重
# params = {}
# for k, v in model1.parameters_dict().items():
#     params[k] = v.shape
#     info = k + ' ' + str(v.shape) + ' ' + str(v.dtype)
#     print(info)




# name_map={}
# for k, v in state_dict.items():
#     info = k + ' ' + str(v.shape) + ' ' + str(v.dtype)
#     info_file.write(info + '\n')
#     print(info)
