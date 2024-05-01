import json
import requests
import os
import time, datetime

# 注意：确保 ollama 服务已启动
# model = "dolphin-phi"  # 根据需要更新模型
root_directory = 'G:\data\\new_tru'


def chat(message, model):
    r = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={"model": model, "messages": [message], "stream": True},
    )

    r.raise_for_status()
    current_output = ""

    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            content = body.get("message", {}).get("content", "")
            current_output += content

        if body.get("done", False):
            return current_output


def find_raw_query_files(root_dir):
    raw_query_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == '_raw_query.json':
                raw_query_files.append(os.path.join(dirpath, filename))
    return raw_query_files


# 调用函数并获得结果
def print_progress_bar(progress, total):
    bar_length = 50
    filled_length = int(bar_length * progress / total)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r[{bar}] {progress}/{total}', end='', flush=True)


def main(inputs, modelname):
    outputs = []
    # 如果输入列表长度大于50，则仅使用前50个元素
    if len(inputs) > 400:
        inputs = inputs[:400]
    print("------共问题数量为：", length)
    i = 1
    for inp in inputs:
        print_progress_bar(i, len(inputs))  # 这个函数假设已经定义了，用来显示进度条
        message = {"role": "user", "content": inp}
        response = chat(message, modelname)  # 这个函数假设已经定义了，用来获取聊天机器人的回应
        outputs.append(response)
        i = i + 1
    return outputs


if __name__ == "__main__":
    # now = datetime.datetime.now()
    # # 设置今天的9点时间
    # today_nine_am = now.replace(hour=9, minute=20, second=0, microsecond=0)
    #
    # # 计算当前时间到今天9点的秒数
    # wait_seconds = (today_nine_am - now).total_seconds()
    #
    # # 等待到今天的9点
    # print(f"等待 {wait_seconds} 秒至今天的9点。")
    # time.sleep(wait_seconds)

    # 在这里放置你想在今天9点执行的任务代码

    length = 0
    inputs = []
    # 打开并读取JSON文件
    # {'dol': 'dolphin-phi'}, {'ge_2': 'gemma:2b'}, {'ge_7': 'gemma:7b'}, {'llama_13': 'llama2:13b'}, {'llama_2': 'llama2:7b'}, {'llava': 'llava'}, {'mistral': 'mistral'}, {'openchat': 'openchat'}, {'qwen': 'qwen'}, {'llama_3': 'llama3:8b'}}
    model_list=[{'llama_3': 'llama3:8b'}]
    for Model in model_list:
        item_name = next(iter(Model.keys()))
        model_name = Model[item_name]
        print('当前模型是：', model_name)
        # 取字典
        raw_query_files_list = find_raw_query_files(root_directory)
        for file_path in raw_query_files_list:
            t1 = time.time()
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                length = len(data)
                for item in data:
                    inputs.append(item['query'])  # 提取'query'的值并添加到列表中

            print("项目进度：", file_path)
            outputs = main(inputs, model_name)
            inputs.clear()
            # 记得清空
            formatted_data = [{"answer": item} for item in outputs]

            # 制造存储
            directory, filename = os.path.split(file_path)

            # 将文件名中的'_raw_query.json'替换为'_raw_answer.json'
            answer_name = '_raw_answer_all_' + item_name + '.json'

            new_filename = filename.replace('_raw_query.json', answer_name)

            # 使用os.path.join()方法将目录路径和新的文件名拼接成新的路径
            store_path = os.path.join(directory, new_filename)

            # 写入到JSON文件
            with open(store_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=4)

            t2 = time.time()
            print("本章耗时：", t2 - t1, "s")
            print("=================================")

# 模拟任务完成
