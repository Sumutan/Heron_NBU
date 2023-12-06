"""
Q:一个文件夹下有很多子文件夹，每个子文件夹中包含很多的.jpg文件，请编写一个python脚本，将每一个子文件夹打包成.zip文件并存放在另一个文件夹中"""
import os
import zipfile

def zip_subfolders(parent_folder, output_folder):
    for foldername, subfolders, filenames in os.walk(parent_folder):
        with zipfile.ZipFile(os.path.join(output_folder, f"{os.path.basename(foldername)}.zip"), "w", zipfile.ZIP_DEFLATED) as zipf:
            for filename in filenames:
                if filename.endswith(".jpg"):
                    file_path = os.path.join(foldername, filename)
                    zipf.write(file_path, arcname=filename)

# 指定要打包的父文件夹路径
parent_folder = "path/to/parent_folder"  # 替换为实际的父文件夹路径

# 指定要保存打包文件的输出文件夹路径
output_folder = "path/to/output_folder"  # 替换为实际的输出文件夹路径

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 调用函数打包子文件夹
zip_subfolders(parent_folder, output_folder)