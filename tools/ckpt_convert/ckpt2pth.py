from collections import OrderedDict

import mindspore as ms
import torch
import numpy as np


def convert_ckpt2pth(ckpt_file):
    state_dict = ms.load_checkpoint(ckpt_file)
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    elif 'module' in state_dict:
        state_dict = state_dict['module']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    else:
        state_dict = state_dict

    new_state_dict = OrderedDict()
    pth_dict = {'model': new_state_dict}
    for k, v in state_dict.items():
        if k == 'scale_sense' or k.startswith('accumulate_') or k.startswith('zeros_') or k.startswith('counter_') or \
                k.startswith('global_step') or k.startswith('adam_m') or k.startswith('adam_v') or \
                k.startswith('current_iterator_step') or k.startswith('last_overflow_iterator_step'):
            continue
        if k.startswith('encoder') or k.startswith('decoder'):
            if k.find('k_bias') >= 0 or k.find('qkv.bias') >= 0:
                continue
        if k.find('norm') >= 0:
            if k.find('gamma') >= 0:
                k = k.replace('gamma', 'weight')
            elif k.find('beta') >= 0:
                k = k.replace('beta', 'bias')

        new_state_dict[k] = torch.tensor(v.asnumpy())
        print(k)

    pth_dict['model'] = new_state_dict
    pth_path = ckpt_file.split('.')[0] + '.pth'
    torch.save(pth_dict, pth_path)


if __name__ == '__main__':
    torch_path = "4_30_model-100_457_acc75.ckpt"

    convert_ckpt2pth(torch_path)