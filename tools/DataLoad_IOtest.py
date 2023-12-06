"""
该文件用于测试cv2与decord库的性能差距
"""
from decord import cpu, VideoReader
import cv2
import numpy as np
import time

VIDEO_PATH = "/home/ma-user/work/vzQDG-vnzFs.mp4"
# VIDEO_PATH ="/home/ma-user/work/9M282d6-CXY.mp4"

FRAME_LIST = [150 + max(0, x * 4 - 1) for x in range(16)]
print(FRAME_LIST, len(FRAME_LIST))

# ================= cv2 test =================

cv2_start_time = time.time()

cap = cv2.VideoCapture(VIDEO_PATH)
frame_num=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_idx, end_idx = 150,150+64-1
video_start_frame = int(start_idx)

cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

frames = []
# 读取指定数量的帧
for j in range(64):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

frames = np.array(frames)

index = np.linspace(0, 64, 16).astype(int)
index = np.clip(index, 0, frames.shape[0] - 1)
new_frames = frames[index, :, :, :]

cv2_end_time = time.time()
cap.release()
print("cv2 read 16 frame take => ", cv2_end_time - cv2_start_time, new_frames.shape)


# cv2_frame_list = []
# cap = cv2.VideoCapture(VIDEO_PATH)
# frame_idx =150
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

# for i in range(64):
#     res, frame = cap.read()
#     if res:
#         cv2_frame_list.append(frame)

# for frame_idx in FRAME_LIST:
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#     res, frame = cap.read()
#     if res:
#         cv2_frame_list.append(frame)
# cv2_end_time = time.time()
# cap.release()
# print("cv2 read 16 frame take => ", cv2_end_time - cv2_start_time, len(cv2_frame_list))
# ================= cv2 test =================

ctx = cpu(0)
# ================= decord test =================
decord_start_time = time.time()
vr = VideoReader(VIDEO_PATH, ctx=ctx)
length=len(vr)
frames = vr.get_batch(FRAME_LIST).asnumpy()
decord_end_time = time.time()
print("decord random read 16 frame take => ", decord_end_time - decord_start_time, frames.shape)
# ================= decord test =================


# ================= decord test =================
decord_start_time = time.time()
vr = VideoReader(VIDEO_PATH, ctx=ctx)
vr.skip_frames(100)
vr.seek(150)
frames=[]
for i in range(16):
    frames.append(vr.next())
decord_end_time = time.time()
print("decord sequence read 16 frame take => ", decord_end_time - decord_start_time, len(frames))
# ================= decord test =================
