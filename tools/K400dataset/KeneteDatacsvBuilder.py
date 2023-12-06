import os
import csv

"用于为K400等分类数据集生成数据索引文件"


def buildDataCsv(root_folder_path: str,buildCLStable=False):
    """
    param root_folder_path：文件夹地址
        like:  E:/tmp/k400/train
               home/k400/train
    运行后会在同级目录下生成对应的csv文件
    在win平台做过测试
    """

    print(f"开始收录{root_folder_path}")
    # 获取所有一级文件夹路径，dirs1为绝对路径，folders1为文件夹名
    dirs1 = [f.path.replace("\\", "/") for f in os.scandir(root_folder_path) if f.is_dir()]
    folders1 = os.listdir(root_folder_path)
    print("num of dirsL1:", len(dirs1))
    assert len(dirs1) == len(folders1), "数量不对等"

    # 获取所有一级文件夹路径，dir2为绝对路径，folders2为文件夹名
    dirs2, folders2 = [], []
    for dir in dirs1:  # 遍历一级文件夹,获取二级文件夹
        dirs2.extend([f.path.replace("\\", "/") for f in os.scandir(dir) if f.is_dir()])
        folders2.extend(os.listdir(dir))
    print("num of dirsL2:", len(dirs2))
    assert len(dirs2) == len(folders2), "数量不对等"

    classnames = [f.replace("_", " ") for f in folders2]  # 标准的类名使用空格而非“_”
    print("class num:", len(classnames))
    assert len(classnames) == 400, "类别数！=400"
    classnames.sort()

    if buildCLStable:
        with open('cls_table.csv', 'w', newline='') as file:
            writer = csv.writer(file)

            # 遍历列表并将每个元素写入CSV文件中
            for item in classnames:
                writer.writerow([item])

    classdic = {}
    clstocsv = []


    for clsnumber, clsname in enumerate(classnames):
        clsname.strip()
        clstocsv.append([clsname, str(clsnumber)])
        classdic[clsname] = clsnumber

    traincsv = []
    for dir2 in dirs2:
        movies = [mov.path.replace("\\", "/") for mov in os.scandir(dir2) if mov.path.endswith('.mp4')]
        cls = classdic[dir2.split("/")[-1].replace("_", " ")]
        for mov in movies:
            traincsv.append([mov, str(cls)])
    print("记载数据数量：", len(traincsv))
    print("示例：", traincsv[:5])

    csvfilename = root_folder_path.split("/")[-1] + '.csv'
    with open(csvfilename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(traincsv)
    print(f'write {csvfilename} finished!')



if __name__ == '__main__':
    root_folder_path = [
        # "E:/tmp/k400/train",
        "E:/tmp/k400/val",
        # "/home/ma-user/work/dataset/k400/train" ,
        # "/home/ma-user/work/dataset/k400/val"
    ]
    for path in root_folder_path:
        buildDataCsv(path,buildCLStable=True)

