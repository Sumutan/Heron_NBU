import csv
def write_list_to_txt(lst, file_path='output.txt'):
    """输入一个列表或string，将列表中元素逐行写入文件中"""
    if type(lst) not in [str ,list]:
        raise TypeError("lst must be a list or string")
    elif type(lst) == str:
        lst = [lst]
    with open(file_path, "w") as file:
        for item in lst:
            file.write(str(item) + "\n")

def read_txt(file_path):
    """读取文件并逐行保存到列表中"""
    lines = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                lines.append(line.strip())  # 去除行尾的换行符并添加到列表中
        return lines
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []

def write_list_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

def read_list_from_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

if __name__ == '__main__':
    lst =[1 ,2 ,3]
    write_list_to_txt(lst)
    lines =read_txt("output.txt")
    print(lines)
