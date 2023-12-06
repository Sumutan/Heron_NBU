import os


def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return len(lines)


def count_python_lines(directory):
    total_lines = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                try:
                    file_path = os.path.join(root, file)
                    lines = count_lines(file_path)
                    total_lines += lines
                    print(f"{file_path}: {lines} lines")
                except:
                    pass
    print(f"Total lines: {total_lines}")


# 指定要统计的目录路径
directory_path = r"C:\Users\13587\Desktop\mae-A\heron_-nbu" #'/home/ma-user/work/code/heron_-nbu'

count_python_lines(directory_path)
