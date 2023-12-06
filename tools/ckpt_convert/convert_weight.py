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
"""Convert checkpoint from torch/facebook"""
import argparse

import mindspore as ms
import torch

def get_keymap_txt(pth_file):
    print("------get_keymap_txt------")
    map_path = pth_file.split('.')[0] + '_key_map.txt'
    map_file = open(map_path, 'w')
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    elif 'module' in state_dict:
        state_dict = state_dict['module']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    
    for k in state_dict:
        new_k = k
        # if new_k.startswith('encoder.'):
        #     new_k = new_k.split('encoder.')[1]
        # else:
        #     continue
        if new_k.startswith('norm.'):
            new_k = new_k.replace('norm.', 'fc_norm.')
        if 'norm' in new_k:
            if 'weight' in new_k:
                new_k = new_k.replace('weight', 'gamma')
            if 'bias' in k:
                new_k = new_k.replace('bias', 'beta')
        map_file.write(k+' '+new_k+'\n')
        # print(k+' '+new_k)
    return map_path

def convert_weight(pth_file="mae_pretrain_vit_base.pth", key_map_path="key_map.txt"):
    """
    convert mae_vit_base_p16 weights from pytorch to mindspore
    pytorch and GPU required.
    """
    ms_ckpt_path = pth_file.split('.')[0] + '.ckpt'
    key_map_dict = {}
    with open(key_map_path, 'r') as f:
        key_map_lines = [s.strip() for s in f.readlines()]
        for line in key_map_lines:
            ckpt_key, model_key = line.split(' ')
            key_map_dict[ckpt_key] = model_key

    ms_ckpt = []
    ckpt = torch.load(pth_file, map_location=torch.device('cpu'))
    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']
    elif 'module' in ckpt:
        state_dict = ckpt['module']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    for k, v in state_dict.items():
        if 'decoder_pos_embed' == k:
            v = v[:, 1:, :]
        if k in key_map_dict:
            k_map = key_map_dict[k]
        else:
            continue

        ms_ckpt.append({'name': k_map, 'data': ms.Tensor(v.numpy())})
        print(k_map)

    ms.save_checkpoint(ms_ckpt, ms_ckpt_path)

def change_keys(pth_file="mae_pretrain_vit_base.pth", key_map_path="key_map.txt"):
    """
    convert mae_vit_base_p16 weights from pytorch to mindspore
    pytorch and GPU required.
    """
    key_map_dict = {}
    with open(key_map_path, 'r') as f:
        key_map_lines = [s.strip() for s in f.readlines()]
        for line in key_map_lines:
            ckpt_key, model_key = line.split(' ')
            key_map_dict[ckpt_key] = model_key

    new_state_dict = {}
    ckpt = torch.load(pth_file, map_location=torch.device('cpu'))
    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']
    elif 'module' in ckpt:
        state_dict = ckpt['module']
    else:
        state_dict = ckpt

    for k, v in state_dict.items():
        if 'decoder_pos_embed' == k:
            v = v[:, 1:, :]
        k_map = key_map_dict[k]
        new_state_dict[k_map] = v
        print(k_map)
    
    ckpt['model_state'] = new_state_dict

    save_file = pth_file.replace('.pth', '_key_changed.pth')
    torch.save(ckpt, save_file)

def show_info(pth_file="mae_pretrain_vit_base.pth"):
    info_path = pth_file.split('.')[0] + '_info.txt'
    info_file = open(info_path, 'w')
    if '.ckpt' in pth_file:
        state_dict = ms.load_checkpoint(pth_file)
    elif '.pth' in pth_file:
        state_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    elif 'module' in state_dict:
        state_dict = state_dict['module']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    
    for k, v in state_dict.items():
        info = k + ' ' + str(v.shape) + ' ' + str(v.dtype)
        info_file.write(info + '\n')
        print(info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mae vit weight convert script")
    parser.add_argument("--torch_path",
                        type=str,
                        default="k400finetune_B.pth",
                        help="The torch checkpoint path.")
    parser.add_argument("--key_map_path",
                        type=str,
                        default="k400finetune_B_map.txt")
    opt = parser.parse_args()

    # get_keymap_txt(opt.torch_path)
    # convert_weight(opt.torch_path, opt.key_map_path)
    # change_keys(opt.torch_path, opt.key_map_path)

    show_info(opt.torch_path)
    keymap_txt = get_keymap_txt(opt.torch_path)
    convert_weight(opt.torch_path, keymap_txt)

