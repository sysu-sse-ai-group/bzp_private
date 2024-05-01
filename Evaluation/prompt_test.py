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


def find_specific_file1(directory, file_name="_rag_real_answer.json"):
    # 遍历指定目录下的所有文件和文件夹
    for dirpath, dirnames, filenames in os.walk(directory):
        # 检查当前目录下的文件名
        for filename in filenames:
            if filename == file_name:
                # 如果文件名匹配，返回绝对路径
                return os.path.join(dirpath, filename)
    return None  # 如果找不到文件，返回None


def deal(file_path1, file_path2, file_path3):
    messages = []
    Socres = []
    inputs = get_input(file_path1)
    actual_outputs = get_output(file_path2)
    retrieval_contexts, expected_outputs = extract_references_from_json(file_path3)
    num = 23
    inputs = inputs[:num]
    actual_outputs = actual_outputs[:num]
    expected_outputs = expected_outputs[:num]
    retrieval_contexts = retrieval_contexts[:num]
    for input, actual_output, expected_output, retrieval_context in zip(inputs, actual_outputs, expected_outputs,
                                                                        retrieval_contexts):
        if retrieval_context is None:
            retrieval_context = 'null'
        else:
            # 确保可以安全地访问 retrieval_context[0]
            if retrieval_context and retrieval_context[0] is not None:
                retrieval_context = retrieval_context[0]
            else:
                # 处理 retrieval_context 是空列表或其第一个元素是 None 的情况
                retrieval_context = 'null'

        user_input = "query:" + input + ", expected_output" + expected_output + ", retrieval_context" + \
                     retrieval_context[0] + ", actual_output:" + actual_output
        if not user_input:
            exit()
        prompt1 = "You are now a judge. I will provide two sentences, one is a query and corresponding answer.The " \
                  "assessment is made on three points: first, whether the question is answered positively. The second " \
                  "point is whether the answer is too long or incomplete. The third point is whether the logical " \
                  "thinking is correct. The first two are more important. The score should range from 0 to 1, " \
                  "accurate to two decimal places. Please give your score. Required format: first " \
                  "give the score just like 'Score: xxx', don't need to explain. Below is our sample, Don't be too " \
                  "strict. "
        prompt2 = "You are now a judge. I will provide 4 sentences, they are a query,a expected answer, " \
                  "a corresponding article context and actual answer." \
                  "You need to judge whether the actual answer meets the Contextual Relevancy based on the three " \
                  "types of information provided above.If actual answer refuse to answer a question which are not " \
                  "sure about, it will also be counted as a very high score, , Random answers should focus on " \
                  "deducting points.. Give an objective and well-founded score from 0 to 1 " \
                  "accurate to two decimal places. Please give your score. Required format: " \
                  "just like 'Score: xxx' in the beginning, don't need to explain. Below is our sample. Be strict."

        prompt3 = "You are now a judge. I will provide 4 sentences, they are a query,a expected answer, " \
                  "a corresponding article context and actual answer." \
                  "Now, you need to determine whether the actual_answer has produced hallucinations, namely factual " \
                  "hallucinations and fidelity hallucinations. For example, it may completely mismatch or even " \
                  "contradict the knowledge in the context, or the generated content may be inconsistent with " \
                  "verifiable real-world facts. If actual answer refuse to answer a question which are not sure about, " \
                  "it will also be counted as a very high score, Random answers should focus on deducting points. " \
                  "Give an objective and well-founded score from 0 to 1 " \
                  "accurate to two decimal places. Please give your score. Required format: " \
                  "just like 'Score: xxx' in the beginning, don't need to explain. Below is our sample. Be objective."

        user_input_final = prompt3 + user_input
        messages.append({"role": "user", "content": user_input_final})
        message = chat(messages)
        if message is None:
            Socres.append("0.80")
            print("Warning1")
        if message['content'] is not None:
            if len(message['content']) > 10:
                Socres.append(message['content'][6:11])
                Socres.append(message['content'][6:11])
            else:
                Socres.append("0.70")
                print("Warning2")
        else:
            Socres.append("0.70")
            print("Warning3")
        messages.clear()
    # 将结果写入JSON文件
    dir_name, file_name = os.path.split(file_path2)
    # 在文件名前添加 '_Score'
    new_file_name = "lm3_ev_Hallu_" + file_name
    # 组合新的文件路径
    store_path = os.path.join(dir_name, new_file_name)
    # 输出新的文件路径
    with open(store_path, 'w', encoding='utf-8') as f:
        json.dump(Socres, f, indent=4)

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
