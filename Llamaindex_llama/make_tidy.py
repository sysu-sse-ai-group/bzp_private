import os
import shutil


def organize_json_files(root_path):
    # 定义文件前缀和目标文件夹的映射
    prefix_to_folder = {
        "vector_answer": "vector_answer",
        "_bm25": "bm25_answer",
        "_raw_answer": "raw_answer"
    }

    # 遍历根目录下的每个文件夹
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        # 检查是否为文件夹
        if os.path.isdir(folder_path):
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                # 检查文件是否为JSON格式
                if file.endswith(".json"):
                    # 根据前缀找到相应的目标文件夹
                    for prefix, target_folder in prefix_to_folder.items():
                        if file.startswith(prefix):
                            # 计算目标文件夹路径
                            target_folder_path = os.path.join(folder_path, target_folder)
                            # 检查目标文件夹是否存在，不存在则创建
                            if not os.path.exists(target_folder_path):
                                os.makedirs(target_folder_path)
                                print(f"Created directory {target_folder_path}")
                            # 计算原始文件的完整路径和目标路径
                            source_file_path = os.path.join(folder_path, file)
                            target_file_path = os.path.join(target_folder_path, file)
                            # 移动文件
                            shutil.move(source_file_path, target_file_path)
                            print(f"Moved '{source_file_path}' to '{target_file_path}'")
                            break  # 跳出循环，继续检查下一个文件

# 使用示例，将 'your_root_directory_path' 替换为您的根目录路径
organize_json_files('G:\data')
