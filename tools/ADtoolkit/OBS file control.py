"""
这是一个用于快速同步OBS、云开发平台、本地文件的工具脚本
"""

# 下载文件/文件夹
# import moxing as mox
#
# obs_path = 'obs://heron-mask/output/finetune_mask_100_surveillance_401cls_clstoken/model-90_583.ckpt'
# workspace_path = '/home/ma-user/work/code/cls_model-90_583.ckpt'
#
# mox.file.copy_parallel(obs_path, workspace_path)   # 从obs上下载 整个 数据集


# 上传文件/文件夹
# import moxing as mox
#
# workspace_path = '/home/ma-user/work/code'
# obs_path = 'obs://heron-nbu/jupyter notebook/code'
#
# mox.file.copy_parallel(workspace_path, obs_path)   # 从obs上下载 整个数据集


# pretrain model convert to finetune model
# import mindspore as ms
#
# ckpt_file="model-800_292.ckpt"
# state_dict = ms.load_checkpoint(ckpt_file)
#
# encoder_ckpt_path = ckpt_file.split('.')[0] + '_encoder.ckpt'
# encoder_state_dict = []
# for k, v in state_dict.items():
#     if k.startswith('encoder.'):
#         print(k)
#         if k.find('norm.') >= 0:
#             k = k.replace('norm.', 'fc_norm.')
#         encoder_state_dict.append({'name': k, 'data': v})
#
# ms.save_checkpoint(encoder_state_dict, encoder_ckpt_path)


# 同步项目文件夹（upload）
import moxing as mox

upload = True  # False means down load
workspace_path = '/home/ma-user/work/code/heron_-nbu'
obs_path = 'obs://heron-nbu/code/heron_-nbu/'

if upload:
    mox.file.copy_parallel(workspace_path, obs_path)
else:
    mox.file.copy_parallel(obs_path, workspace_path)
