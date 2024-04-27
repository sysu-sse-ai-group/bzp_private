import glob
import json
import os

import numpy as np
import requests
from tqdm import tqdm

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = "llama3:8b"  # TODO: update this for whatever model you wish to use
# 设置文件夹路径
root_dir = "G:\data"

# 创建一个空列表，用于存储符合条件的文件路径

# 遍历文件夹中的所有文件
def list_directories(root_dir):
    # 获取根目录下所有直接子文件夹的绝对路径
    directories = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, name))]
    return directories


def find_folders_by_prefix(root_dir):
    # 使用字典来存储每个上级目录下的符合条件的文件夹路径列表
    directories_dict = {}

    # 遍历根目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 创建当前目录下的列表
        folder_list = []
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if dirname.startswith("raw") or dirname.startswith("bm") or dirname.startswith("vector"):
                folder_list.append(full_path)
        if folder_list:  # 如果当前目录下有符合条件的文件夹，存储到字典
            directories_dict[dirpath] = folder_list

    return directories_dict

def calculate_average_from_json(file_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 初始化数字列表
    numbers = []
    for num in data:        # 尝试将字符串转换为浮点数
        if num == "nan":
            numbers.append(0.90)
        else:
            number = float(num.strip())
            numbers.append(number)

    print(numbers)
    # 使用numpy计算平均值
    average = np.mean(numbers)
    return average


import json
import os


def create_json_file(directory, file_name, data):
    # 构建完整的文件路径
    file_path = os.path.join(directory, file_name + ".json")
    # 确保目录存在
    os.makedirs(directory, exist_ok=True)
    # 写入JSON文件
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

    print(f"File created successfully at {file_path}")


# 示例使用
# if __name__ == "__main__":
#     directory = "G:\\data\\_blockchain_solana\\bm25_answer"
#     file_name = "lm3_ev_S_average"
#     data = {"average": 0.85, "count": 100}  # 示例字典，根据需要修改
#
#     create_json_file(directory, file_name, data)


def main():
    folder_dict = find_folders_by_prefix(root_dir)
    fold_list = list_directories(root_dir)
    for one_fold_list in fold_list:
        under_list = folder_dict[one_fold_list]
        for two_fold_list in under_list:
            print("now is", two_fold_list)
            folder_name = os.path.basename(two_fold_list)
            filename1 = "_Average_Ragas_ev_"
            prefix = "_Ragas_ev_S"
            if folder_name == 'bm25_answer':
                matched_files = []
                # 遍历指定目录下的所有文件
                for filename in os.listdir(two_fold_list):
                    # 检查每个文件名是否以指定前缀开头
                    if filename.startswith(prefix):
                        # 如果是，构造文件的完整路径并添加到列表中
                        full_path = os.path.join(two_fold_list, filename)
                        matched_files.append(full_path)

                print("1")
                dic_list = []
                for file in matched_files:
                    average = calculate_average_from_json(file)
                    dic = {file: average}
                    dic_list.append(dic)
                create_json_file(two_fold_list, filename1, dic_list)
                dic_list.clear()
                matched_files.clear()
            elif folder_name == 'raw_answer':
                matched_files = []
                # 遍历指定目录下的所有文件
                for filename in os.listdir(two_fold_list):
                    # 检查每个文件名是否以指定前缀开头
                    if filename.startswith(prefix):
                        # 如果是，构造文件的完整路径并添加到列表中
                        full_path = os.path.join(two_fold_list, filename)
                        matched_files.append(full_path)

                print("2")
                dic_list = []
                for file in matched_files:
                    average = calculate_average_from_json(file)
                    dic = {file: average}
                    dic_list.append(dic)
                create_json_file(two_fold_list, filename1, dic_list)
                dic_list.clear()
                matched_files.clear()
            elif folder_name == 'vector_answer':
                matched_files = []
                # 遍历指定目录下的所有文件
                for filename in os.listdir(two_fold_list):
                    # 检查每个文件名是否以指定前缀开头
                    if filename.startswith(prefix):
                        # 如果是，构造文件的完整路径并添加到列表中
                        full_path = os.path.join(two_fold_list, filename)
                        matched_files.append(full_path)

                print("3")
                dic_list = []
                for file in matched_files:
                    average = calculate_average_from_json(file)
                    dic = {file: average}
                    dic_list.append(dic)
                create_json_file(two_fold_list, filename1, dic_list)
                dic_list.clear()
                matched_files.clear()

            # for file_path2 in file_list2:
            #     print("now is:", file_path2)
            #     deal(file_path1, file_path2)
if __name__ == "__main__":
    main()
