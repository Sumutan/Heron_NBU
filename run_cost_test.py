from src.video_dataset_new import MaskMapGenerator
import numpy as np
import time

# 定义要计时的函数

if __name__ == '__main__':
    # 假设有一个形状为 [2, 3, 100, 100, 3] 的 RGB 图片数组
    rgb_images = np.random.randint(0, 256, size=(1, 16, 400, 400, 3), dtype=np.uint8)

    # 将 RGB 图片转换为灰度图
    MaskMapGenerator=MaskMapGenerator(testMode=True)

    # gray_images = MaskMapGenerator.rgb_to_gray_np(rgb_images)
    # gray_images = MaskMapGenerator.rgb_to_gray_np_cv2(rgb_images)

    # 时间测试
    start_time = time.time()
    # my_function()
    for i in range(1):
        gray_images = MaskMapGenerator.rgb_to_gray_np(rgb_images)
    end_time = time.time()
    execution_time = end_time - start_time
    print("执行时间为: ", execution_time, "秒")

    # 时间测试
    start_time = time.time()
    # my_function()
    for i in range(10):
        gray_images = MaskMapGenerator.rgb_to_gray_np_cv2(rgb_images)
    end_time = time.time()
    execution_time = end_time - start_time
    print("执行时间为: ", execution_time, "秒")

    # 打印灰度图像数组的形状
    print(gray_images.shape)

