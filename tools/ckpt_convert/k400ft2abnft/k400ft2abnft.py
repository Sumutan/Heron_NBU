"""
transform the ckpt from 400 cls finetune to 2cls finetune for Abnormal Detection by remove cls_head
"""

import os
import mindspore as ms


def load_ckpt(pth_file):
    if '.ckpt' in pth_file:
        state_dict = ms.load_checkpoint(pth_file)
    # elif '.pth' in pth_file:
    #     state_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    else:
        raise ValueError("pth_file must be .ckpt or .pth")
    return state_dict

def show_info(pth_file):
    info_path = pth_file.split('.')[0] + '_info.txt'
    info_path = os.path.basename(info_path)  # 写在当前目录下
    info_file = open(info_path, 'w')

    state_dict=load_ckpt(pth_file)

    # print(state_dict.keys())
    # for key in state_dict.keys():
    #     info_file.write(key + '\n')
    # info_file.close()

    for k, v in state_dict.items():
        info = k + ' ' + str(v.shape) + ' ' + str(v.dtype)
        info_file.write(info + '\n')
        print(info)
    info_file.close()

def del_head(pth_file="mae_pretrain_vit_base.pth",):
    """
    remove head.weight and head.bias
    """
    state_dict = load_ckpt(pth_file)
    ms_ckpt=[]

    for k, v in state_dict.items():
        if 'head' not in k:
            ms_ckpt.append({'name': k, 'data': v})
            print(k)

    output_file = os.path.join(os.path.dirname(pth_file)+'/2cls_ckpt',os.path.basename(pth_file).split('.')[0]+'_2cls.ckpt')
    ms.save_checkpoint(ms_ckpt, output_file)

if __name__ == '__main__':
    # set path
    ckpt_path = '/home/ma-user/work/ckpt'
    ckpt_name = 'finetune_frame_with_depth_401l-100_734.ckpt'
    aim_ckpt_path = '/home/ma-user/work/ckpt/2cls_ckpt'

    # show struct info
    ckpt = os.path.join(ckpt_path, ckpt_name)
    show_info(pth_file=ckpt)
    del_head(ckpt)
