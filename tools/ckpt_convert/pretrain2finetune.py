import mindspore as ms

ckpt_file="/home/ma-user/work/ckpt/8-15_pretrain_random_on_surveillance_20w_800e-800_372.ckpt"
state_dict = ms.load_checkpoint(ckpt_file)

encoder_ckpt_path = ckpt_file.split('.')[0] + '_encoder.ckpt'
encoder_state_dict = []
for k, v in state_dict.items():
    if k.startswith('encoder.'):
        print(k)
        if k.find('norm.') >= 0:
            k = k.replace('norm.', 'fc_norm.')
        encoder_state_dict.append({'name': k, 'data': v})

ms.save_checkpoint(encoder_state_dict, encoder_ckpt_path)