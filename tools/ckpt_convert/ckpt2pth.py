# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Convert .pth from mindspore(.ckpt)"""

from collections import OrderedDict

import mindspore as ms
import torch
import numpy as np


def get_statedict(ckpt_file):
    if ckpt_file.endswith('.ckpt'):
        state_dict = ms.load_checkpoint(ckpt_file)
    elif ckpt_file.endswith('.pth'):
        state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
    else:
        raise ValueError('Not support ckpt file: {}'.format(ckpt_file))

    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    elif 'module' in state_dict:
        state_dict = state_dict['module']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    else:
        state_dict = state_dict

    return state_dict

def get_keymap_txt(pth_file):
    """get keymap (ckpt->pth)"""
    print("------get_keymap_txt------")
    map_path = pth_file.split('.ckpt')[0].split('.pth')[0] + '_key_map.txt'


    map_file = open(map_path, 'w')
    state_dict = get_statedict(pth_file)

    for k in state_dict:
        new_k = k
        if new_k.startswith('fc_norm.'):
            new_k = new_k.replace('fc_norm.', 'norm.')
        if 'norm' in new_k:
            if 'gamma' in new_k:
                new_k = new_k.replace('gamma', 'weight')
            if 'beta' in k:
                new_k = new_k.replace('beta', 'bias')

        # if 'patch_embed' not in new_k:
        #     new_k='encoder.'+new_k
        new_k='encoder.'+new_k


        map_file.write(k + ' ' + new_k + '\n')
        # print(k+' '+new_k)

    return map_path

def convert_weight(ckpt_file="ViT-B_VideoMAE_ep1600_k400.ckpt", key_map_path="key_map.txt"):
    """
    convert mae_vit_base_p16 weights from mindspore to pytorch (ckpt->pth)
    """
    key_map_dict = {}
    with open(key_map_path, 'r') as f:
        key_map_lines = [s.strip() for s in f.readlines()]
        for line in key_map_lines:
            old_key, new_key = line.split(' ')
            key_map_dict[old_key] = new_key

    state_dict=get_statedict(ckpt_file)

    # pth_ckpt = []
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k in key_map_dict.keys():
            k_map = key_map_dict[k]
        else:
            continue
        new_state_dict[k_map] = torch.tensor(v.asnumpy())
        # new_state_dict.append({'name': k_map, 'data': torch.tensor(v.asnumpy())})

        print(k_map)

    pth_ckpt = {'model': new_state_dict}
    torch_ckpt_path = ckpt_file.split('.ckpt')[0] + '.pth'
    torch.save(pth_ckpt, torch_ckpt_path)
    return torch_ckpt_path


    # for k, v in state_dict.items():
    #     if k in key_map_dict:
    #         k_map = key_map_dict[k]
    #     else:
    #         continue
    #
    #     pth_ckpt.append({'name': k_map, 'data': ms.Tensor(v.numpy())})
    #     print(k_map)
    #
    # # ms.save_checkpoint(ms_ckpt, ms_ckpt_path)
    # torch.save(pth_ckpt, torch_ckpt_path)

def convert_ckpt2pth(ckpt_file):
    """似乎是半成品"""
    state_dict=get_statedict(ckpt_file)

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
    pth_path = ckpt_file.split('.ckpt')[0] + '.pth'
    torch.save(pth_dict, pth_path)

def show_info(pth_file="mae_pretrain_vit_base.pth"):
    """将输出有关模型权重的信息并将其写入一个文本文件"""
    info_path = pth_file.split('.ckpt')[0].split('.pth')[0] + '_info.txt'
    info_file = open(info_path, 'w')

    state_dict = get_statedict(pth_file)

    for k, v in state_dict.items():
        info = k + ' ' + str(v.shape) + ' ' + str(v.dtype)
        info_file.write(info + '\n')
        print(info)

if __name__ == '__main__':
    ckpt_path = "./k400finetune/ViT-B_VideoMAE_ep1600_k400.ckpt"

    # convert_ckpt2pth(ckpt_path)


    show_info(ckpt_path)
    keymap_txt = get_keymap_txt(ckpt_path)
    pth_ckpt_path = convert_weight(ckpt_path, keymap_txt)
    show_info(pth_ckpt_path)
