import openai
from deepeval import evaluate
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCase
import os

os.environ['OPENAI_API_BASE'] = "https://api.chatanywhere.cn/v1"
os.environ['OPENAI_API_KEY'] = 'sk-BRIGHrNUlXTmWZxfUn7aOBfMlLh7LicIpRohEyMgDJYfo7q4'

import glob
import json
import os

import requests
from tqdm import tqdm
# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = "llama3:8b"  # TODO: update this for whatever model you wish to use
# 设置文件夹路径
root_dir = "G:\data"

begin = 'vector'
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

def chat(messages):

    r = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True},
    )

    r.raise_for_status()
    output = ""

    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            # the response streams one token at a time, print that as we receive it
            # print(content, end="", flush=True)

        if body.get("done", False):
            message["content"] = output
            return message


# 用来存储所有查询的列表

# 读取JSON文件
def get_input(file_path1):
    inputs = []
    with open(file_path1, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 加载JSON数据

        # 遍历列表中的每个字典
        for item in data:
            # 检查每个字典中是否存在'query'键
            if 'query' in item:
                inputs.append(item['query'])  # 将'query'的值添加到inputs列表
    return inputs

def get_output(file_path2):
    actual_outputs = []
    with open(file_path2, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 加载JSON数据

        # 遍历列表中的每个字典
        for item in data:
            if 'answer' in item:
                actual_outputs.append(item['answer'])  # 将'query'的值添加到inputs列表
    return actual_outputs

def find_specific_file(directory, file_name="_raw_query.json"):
    # 遍历指定目录下的所有文件和文件夹
    for dirpath, dirnames, filenames in os.walk(directory):
        # 检查当前目录下的文件名
        for filename in filenames:
            if filename == file_name:
                # 如果文件名匹配，返回绝对路径
                return os.path.join(dirpath, filename)
    return None  # 如果找不到文件，返回None

def find_specific_file1(directory, file_name="_rag_real_answer.json"):
    # 遍历指定目录下的所有文件和文件夹
    for dirpath, dirnames, filenames in os.walk(directory):
        # 检查当前目录下的文件名
        for filename in filenames:
            if filename == file_name:
                # 如果文件名匹配，返回绝对路径
                return os.path.join(dirpath, filename)
    return None  # 如果找不到文件，返回None

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

def deal(file_path1, file_path2, file_path3):
    Socres = []
    inputs = get_input(file_path1)
    actual_outputs = get_output(file_path2)
    retrieval_contexts, expected_outputs = extract_references_from_json(file_path3)
    num = 8
    inputs = inputs[:num]
    actual_outputs = actual_outputs[:num]
    expected_outputs = expected_outputs[:num]
    retrieval_contexts = retrieval_contexts[:num]

    for input, actual_output, expected_output, retrieval_context in zip(inputs, actual_outputs, expected_outputs, retrieval_contexts):
        metric = RagasMetric(threshold=0.5, model="gpt-3.5-turbo")
        if retrieval_context is None:
            retrieval_context = "null"
        test_case = LLMTestCase(
            input=input,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        metric.measure(test_case)
        print(metric.score)
        score = str(metric.score)
        print(score)
        Socres.append(score)
    # 将结果写入JSON文件
    dir_name, file_name = os.path.split(file_path2)
    # 在文件名前添加 '_Score'
    new_file_name = "_Ragas_ev_S_" + file_name
    # 组合新的文件路径
    store_path = os.path.join(dir_name, new_file_name)
    print(store_path)
    # 输出新的文件路径
    with open(store_path, 'w', encoding='utf-8') as f:
        json.dump(Socres, f, indent=4)
        print("success")

    print("Results saved to ", store_path)

def main():
    folder_dict = find_folders_by_prefix(root_dir)
    fold_list = list_directories(root_dir)
    for one_fold_list in fold_list:
        under_list = folder_dict[one_fold_list]
        file_path1 = find_specific_file(one_fold_list)
        file_path3 = find_specific_file1(one_fold_list)
        for two_fold_list in under_list:
            print("now is", two_fold_list)
            folder_name = os.path.basename(two_fold_list)
            if folder_name == 'bm25_answer':
                prefix = "_bm"
                matched_files = []
                # 遍历指定目录下的所有文件
                for filename in os.listdir(two_fold_list):
                    # 检查每个文件名是否以指定前缀开头
                    if filename.startswith(prefix):
                        # 如果是，构造文件的完整路径并添加到列表中
                        full_path = os.path.join(two_fold_list, filename)
                        matched_files.append(full_path)

                for file_path2 in tqdm(matched_files, desc="Processing files"):
                    deal(file_path1, file_path2, file_path3)
                matched_files.clear()
            elif folder_name == 'raw_answer':
                prefix = "_raw"
                matched_files = []
                # 遍历指定目录下的所有文件
                for filename in os.listdir(two_fold_list):
                    # 检查每个文件名是否以指定前缀开头
                    if filename.startswith(prefix):
                        # 如果是，构造文件的完整路径并添加到列表中
                        full_path = os.path.join(two_fold_list, filename)
                        matched_files.append(full_path)
                for file_path2 in tqdm(matched_files, desc="Processing files"):
                    deal(file_path1, file_path2, file_path3)
                matched_files.clear()
            elif folder_name == 'vector_answer':
                prefix = "vector"
                matched_files = []
                # 遍历指定目录下的所有文件
                for filename in os.listdir(two_fold_list):
                    # 检查每个文件名是否以指定前缀开头
                    if filename.startswith(prefix):
                        # 如果是，构造文件的完整路径并添加到列表中
                        full_path = os.path.join(two_fold_list, filename)
                        matched_files.append(full_path)
                for file_path2 in tqdm(matched_files, desc="Processing files"):
                    deal(file_path1, file_path2, file_path3)
                matched_files.clear()

            # for file_path2 in file_list2:
            #     print("now is:", file_path2)
            #     deal(file_path1, file_path2)

if __name__ == "__main__":
    main()






# or evaluate test cases in bulk
