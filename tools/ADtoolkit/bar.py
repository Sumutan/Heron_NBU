from tqdm import tqdm
import time
from random import random, randint


# 传入可迭代对象
# for i in tqdm(range(10)):
#     # 模拟任务的延迟
#     time.sleep(0.5)

# 手动更新
# bar = tqdm(total=100)
# for i in range(100):
#     time.sleep(0.5)
#     bar.update(1)
#     # 进阶设置
#     # tqdm.write("test")  # 替代print输出
#     bar.set_description("Processing %s" % i)  # 设置进度条左边显示的信息
#     bar.set_postfix(loss=random(), gen=randint(1, 999))  # 设置进度条右边显示的信息
# bar.close()  # 记得关闭

def run(x):
    bar = tqdm(total=100)
    for i in range(10):
        time.sleep(0.5)
        bar.update(10)
        # 进阶设置
        # tqdm.write("test")  # 替代print输出
        bar.set_description("Processing %s" % x)  # 设置进度条左边显示的信息
        bar.set_postfix(loss=random(), gen=randint(1, 999))  # 设置进度条右边显示的信息
    bar.close()  # 记得关闭

# 多进程下使用:https://blog.csdn.net/qq_34914551/article/details/119451639
# pbar = tqdm(total=len(data))
# pbar.set_description('Sleep')
# update = lambda *args: pbar.update()
#
# n_proc = 5
# pool = mp.Pool(n_proc)
# for d in data:
# 	pool.apply_async(test_func, (d,), callback=update)
# pool.close()
# pool.join()
