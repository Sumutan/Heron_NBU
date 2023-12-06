import os.path
import pickle


def is_abnormal(video_pth):  # 绝对路径
    pkl_save_pth = '/home/ma-user/work/dataset/NB_tech/NBUabnormal_snippets/'

    videoname = os.path.basename(video_pth).split('.')[0]  # 不带后缀的文件名
    [vtype,scence,index] = videoname.split('_')  # need check  , index:1,2,3...
    pkl_name=vtype+'_'+scence

    pkl_pth = os.path.join(pkl_save_pth, pkl_name + '.pickle')

    with open(pkl_pth, 'rb') as file:
        gt = pickle.load(file)[pkl_name]  # gt:list contain frame level label
        begin_frame = (int(index) - 1) * 64  # every video have 64 frames
        gt_snippets = gt[begin_frame:begin_frame + 64]

    if max(gt_snippets) > 0.5:
        return True
    else:
        return False


# 打开.pickle文件
# with open('/home/ma-user/work/dataset/NB_tech/NBUabnormal_snippets/falling_10-1.pickle', 'rb') as file:
#     # 使用pickle.load()方法加载对象
#     loaded_object = pickle.load(file)


if __name__ == '__main__':
    testvideo='/home/ma-user/work/dataset/NB_tech/NBUabnormal_snippets/scene13/fighting_13-1/fighting_13-1_114.mp4'
    print(is_abnormal(testvideo))
    pass