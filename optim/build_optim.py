import json
import math

import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor, nn, ops

from lr.lr_schedule import LearningRateWiseLayer

from .optimizer import AdamWeightDecayOp


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token"):
        return 0
    elif "pos_embed" in var_name:
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1

class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))

def get_parameter_groups(model, base_lr, weight_decay=1e-5, skip_list=(), skip_keywords=(), get_num_layer=None, get_layer_scale=None):
    """get finetune param groups"""
    parameter_group_names = {}
    parameter_group_vars = {}

    for param in model.trainable_params():
        name = param.name
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or \
            check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": LearningRateWiseLayer(base_lr, scale),
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def create_optimizer(args, model, lr):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    num_layers = args.depth
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        args.logger.info(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        args.logger.info(f'No weight decay keywords: {skip_keywords}')

    group_parameters = get_parameter_groups(model, 
                                            lr, 
                                            weight_decay=weight_decay, 
                                            skip_list=skip, 
                                            skip_keywords=skip_keywords,
                                            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
                                            get_layer_scale=assigner.get_scale if assigner is not None else None)
    
    opt_args = dict(learning_rate=args.start_learning_rate, weight_decay=weight_decay)
    if hasattr(args, 'beta1') and args.beta1 is not None:
        opt_args['beta1'] = args.beta1
    if hasattr(args, 'beta2') and args.beta2 is not None:
        opt_args['beta2'] = args.beta2
    if hasattr(args, 'eps') and args.eps is not None:
        opt_args['eps'] = args.eps
    
    args.logger.info(f"optimizer settings: {opt_args}")

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    assert opt_lower == 'adamw'
    
    optimizer = nn.AdamWeightDecay(group_parameters, **opt_args)
    return optimizer

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin