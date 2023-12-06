import argparse
import pickle
import sys
import os
import os.path as osp
import glob
from multiprocessing import Pool
import cv2
import pickle

with open('/home/ma-user/work/dataset/XD-Violence/test/gt-violence-dic_tevad_test.pickle','rb') as file:
    gt_dict = pickle.load(file)

def dump_frames(vid_item):
    full_path, vid_path, vid_id = vid_item
    print(vid_path)
    #print(full_path)
    vid_name = vid_path.split("/")

    # out_full_path = osp.join("/home/ma-user/work/dataset/XD-Violence/test/frames", vid_name[0].replace(".mp4",""))
    # out_full_path = osp.join("/home/ma-user/work/dataset/XD-Violence/train/frames", vid_name[0].replace(".mp4",""))
    out_full_path = full_path.replace(".mp4","").replace("/videos","/frames")

    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path)
   
    vr = cv2.VideoCapture(full_path)
    videolen = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    if "XD-Violence/train/" in out_full_path:
        videolen_true= videolen                            # TEVAD train set
    elif "XD-Violence/test/" in out_full_path:
        videolen_true = len(gt_dict[vid_name[0][:-4]])   # TEVAD test set
    else:
        raise RuntimeError("XD-Violence/train/ or XD-Violence/test/ not in out_full_path")
        exit()

    for i in range(videolen_true):
        ret, frame = vr.read()
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is not None:
            cv2.imwrite('{}/img_{:08d}.jpg'.format(out_full_path, i + 1), img)
        else:
            print('[Warning] length inconsistent!'
                  'Early stop with {} out of {} frames'.format(i + 1, videolen))
            break
    print('full_path={} vid_name={} num_frames={} dump_frames done'.format(full_path, vid_name, videolen))
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--src_dir',default="/home/ma-user/work/dataset/XD-Violence/train/videos", type=str)
    parser.add_argument('--out_dir',default="/home/ma-user/work/dataset/XD-Violence/train/frames",type=str)
    parser.add_argument('--level', type=int,
                        choices=[1, 2],
                        default=2)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument("--out_format", type=str, default='dir',
                        choices=['dir', 'zip'], help='output format')
    parser.add_argument("--ext", type=str, default='mp4',
                        choices=['avi', 'mp4'], help='video file extensions')
    parser.add_argument("--resume", action='store_true', default=False,
                        help='resume optical flow extraction '
                        'instead of overwriting')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)
    print("args.src_dir:",args.src_dir)
    fullpath_list = glob.glob(args.src_dir + '/*' )
    done_fullpath_list = glob.glob(args.out_dir + '/*')
    print('Total number of videos found: ', len(fullpath_list))
    if args.resume:
        fullpath_list = set(fullpath_list).difference(set(done_fullpath_list))
        fullpath_list = list(fullpath_list)
        print('Resuming. number of videos to be done: ', len(fullpath_list))

    vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))
    # rm_list = os.listdir("/mnt/disk2/tcc/dataset/XD-Violence/test/frames")
    # vid_list = [x for x in vid_list if x[:-4] not in rm_list]
    #vid_list = ['v=ROrpKx3aIjA__#1_label_G-0-0']

    pool = Pool(180)
    pool.map(dump_frames, zip(fullpath_list, vid_list, range(len(vid_list))))
