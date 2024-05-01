import json
import os


def read_and_process_json_file(file_path):
    # 读取并处理JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    result = []
    if 'examples' in data:
        examples = data['examples']
        for item in examples:
            new_entry = {}
            if 'query' in item:
                new_entry['query'] = item['query']
            if 'reference_contexts' in item:
                new_entry['reference_contexts'] = item['reference_contexts']
            if 'reference_answer' in item:
                new_entry['reference_answer'] = item['reference_answer']
            if new_entry:
                result.append(new_entry)
    return result


def save_data_to_json(result, output_file_path):
    # 保存数据到JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)


def process_all_folders(root_directory):
    # 遍历根目录下的所有文件夹
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(folder_path):
            json_file_path = os.path.join(folder_path, '_rag_dataset.json')
            if os.path.exists(json_file_path):
                # 读取和处理JSON文件
                extracted_data = read_and_process_json_file(json_file_path)
                # 定义输出文件路径
                output_file_path = os.path.join(folder_path, '_rag_real_answer.json')
                # 保存处理后的数据到新的JSON文件
                save_data_to_json(extracted_data, output_file_path)
                print(f"Processed data saved to {output_file_path}")


# 使用函数
root_directory = 'G:\data'  # 请将此路径替换为你的根目录路径
process_all_folders(root_directory)
