"""
用于删除百万级小文件
"""
import os
import timeit
from tqdm import tqdm


def main():
    acc=0
    bar = tqdm()
    for pathname,dirnames,filenames in os.walk('/home/ma-user/work/dataset/ucf_crime/frames_depth'):
        for filename in filenames:
            file=os.path.join(pathname,filename)
            os.remove(file)
            acc+=1
            if acc %100000==0:
                bar.update(100000)
    print(f"delete {acc} file ")
    bar.close()  # 记得关闭

if __name__ == '__main__':
    t = timeit.Timer('main()', 'from __main__ import main')
    print(t.timeit(1))