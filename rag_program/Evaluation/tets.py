import os
import time
import json
import os

# # 原始文件路径
# file_path2 = "G:\\data\\_blockchain_solana\\raw_answer\\_raw_answer_ge_2.json"
# # 分解原始路径为目录和文件名
# dir_name, file_name = os.path.split(file_path2)
# # 在文件名前添加 '_Score'
# new_file_name = "_Score" + file_name
# # 组合新的文件路径
# store_path = os.path.join(dir_name, new_file_name)
# # 输出新的文件路径
# print("New file path:", store_path)

# JSON文件的路径
# file_path1 = "G:\data\_blockchain_solana\_raw_query.json"
# file_path2 = "G:\data\_blockchain_solana\\raw_answer\_raw_answer_ge_2.json"
# # 用来存储所有查询的列表
# inputs = []
# outputs = []
# # 读取JSON文件
# with open(file_path1, 'r', encoding='utf-8') as file:
#     data = json.load(file)  # 加载JSON数据
#
#     # 遍历列表中的每个字典
#     for item in data:
#         # 检查每个字典中是否存在'query'键
#         if 'query' in item:
#             inputs.append(item['query'])  # 将'query'的值添加到inputs列表
#
# with open(file_path2, 'r', encoding='utf-8') as file:
#     data = json.load(file)  # 加载JSON数据
#
#     # 遍历列表中的每个字典
#     for item in data:
#         # 检查每个字典中是否存在'query'键
#         if 'answer' in item:
#             outputs.append(item['answer'])  # 将'query'的值添加到inputs列表
#
# inputs = inputs[:10]
# outputs = outputs[:10]
#
# # 输出结果列表
# print(len(inputs))
# print(len(outputs))
# print(inputs)
# print(outputs)

# def find_raw_query_files(root_dir):
#     raw_query_files = []
#     for dirpath, _, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename == '_raw_query.json':
#                 raw_query_files.append(os.path.join(dirpath, filename))
#     return raw_query_files
#
# # 指定要搜索的根目录
# root_directory = 'G:\data'
# B = []
# # 调用函数并获得结果
# raw_query_files_list = find_raw_query_files(root_directory)
# print(raw_query_files_list)
#
# for file_path in raw_query_files_list:
#     inputs = []
#     t1 = time.time()
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#         length = len(data)
#         for item in data:
#             inputs.append(item['query'])  # 提取'query'的值并添加到列表中
#         print(len(inputs))
#         print(inputs)
# for a in raw_query_files_list:
#     # 使用os.path.split()方法分割文件路径
#     directory, filename = os.path.split(a)
#
#     # 将文件名中的'_raw_query.json'替换为'_raw_answer.json'
#     new_filename = filename.replace('_raw_query.json', '_raw_answer.json')
#
#     # 使用os.path.join()方法将目录路径和新的文件名拼接成新的路径
#     new_file_path = os.path.join(directory, new_filename)
#     B.append(new_file_path)
#
#
# # 打印结果
# print(B)
#
# import time
#
#
# def print_progress_bar(progress, total):
#     bar_length = 50
#     filled_length = int(bar_length * progress / total)
#     bar = '#' * filled_length + '-' * (bar_length - filled_length)
#     print(f'\r[{bar}] {progress}/{total}', end='', flush=True)
#
# # 假设任务数量为10
# total_tasks = 10
#
# # 模拟任务完成
# for i in range(1, total_tasks + 1):
#     # 每完成一个任务，调用打印进度条函数
#     print_progress_bar(i, total_tasks)
#     # 模拟任务执行时间
#     # 这里可以替换为实际任务的代码
#     import time
#     time.sleep(0.5)
#
# print()  # 完成后换行，确保进度条后面不会有其他内容

# model_list = ['dolphin-phi', 'gemma:2b', 'gemma:7b', 'llama:13b', 'llama:7b', 'llava', 'mistral', 'opencaht', 'qwen']
# dict_list = [{"name": model} for model in model_list]
# print(dict_list)

# model_list=[{'dol': 'dolphin-phi'}, {'ge_2': 'gemma:2b'}, {'ge_7': 'gemma:7b'}, {'llama_13': 'llama:13b'}, {'llama_2': 'llama:7b'}, {'llava': 'llava'}, {'mistral': 'mistral'}, {'openchat': 'opencaht'}, {'qwen': 'qwen'}]
# for Model in model_list:
#     item_name = next(iter(Model.keys()))
#     model_name = Model[item_name]
#     print('当前模型是：', model_name)
#     answer_name = '_raw_answer_' + item_name + '.json'
#     print(answer_name)
# import torch
# from transformers import AutoModelForCausalLM
#
# # 打印 PyTorch 的版本
# print("PyTorch Version:", torch.__version__)
#
# # 检查CUDA是否可用
# if torch.cuda.is_available():
#     print("CUDA is available.")
# else:
#     print("CUDA is not available. Model will run on CPU.")
#
# # 设置模型路径
# model_path = "G:\model\Phi3"
#
# # 加载模型
# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
#
# # 将模型移动到GPU
# if torch.cuda.is_available():
#     model.to("cuda")
#     print("Model moved to GPU.")
# else:
#     print("Model running on CPU.")
#
# if torch.cuda.is_available():
#     model.cuda()
# else:
#     print("CUDA is not available. Model is running on CPU.")

import json


def extract_references_from_json(file_path):
    # 读取并解析JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 确保数据是一个列表
    if not isinstance(data, list):
        raise ValueError("JSON file does not contain a list")

    # 初始化两个结果列表
    contexts = []
    answers = []

    # 遍历列表中的每个字典项
    for item in data:
        # 提取reference_contexts和reference_answer
        context = item.get('reference_contexts', 'No context available')
        answer = item.get('reference_answer', 'No answer available')
        # 将提取的数据添加到对应的列表
        contexts.append(context)
        answers.append(answer)

    return contexts, answers


# 示例使用
if __name__ == "__main__":
    file_path = "G:\data\_covid19_paper\_rag_real_answer.json"
    contexts, answers = extract_references_from_json(file_path)
    print("Contexts:", contexts)
    print("Answers:", answers)





