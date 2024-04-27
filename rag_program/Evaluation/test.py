from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
import os
import json
from tqdm import tqdm

# 设置环境变量
os.environ['OPENAI_API_BASE'] = "https://api.chatanywhere.cn/v1"
os.environ['OPENAI_API_KEY'] = 'sk-BRIGHrNUlXTmWZxfUn7aOBfMlLh7LicIpRohEyMgDJYfo7q4'

# 定义输入和输出的列表
# JSON文件的路径
file_path1 = "G:\data\_covid19_paper\_raw_query.json"
file_path2 = "G:\data\_covid19_paper\\bm25_answer\_bm25_answer_qwen.json"
# 用来存储所有查询的列表
inputs = []
actual_outputs = []
# 读取JSON文件
with open(file_path1, 'r', encoding='utf-8') as file:
    data = json.load(file)  # 加载JSON数据

    # 遍历列表中的每个字典
    for item in data:
        # 检查每个字典中是否存在'query'键
        if 'query' in item:
            inputs.append(item['query'])  # 将'query'的值添加到inputs列表

with open(file_path2, 'r', encoding='utf-8') as file:
    data = json.load(file)  # 加载JSON数据

    # 遍历列表中的每个字典
    for item in data:
        # 检查每个字典中是否存在'query'键
        if 'answer' in item:
            actual_outputs.append(item['answer'])  # 将'query'的值添加到inputs列表

inputs = inputs[:20]
actual_outputs = actual_outputs[:20]

# 创建评估度量
metric = AnswerRelevancyMetric(
    threshold=0.5,
    model="gpt-3.5-turbo",
    include_reason=True
)

# 用来存储评估结果的列表
results = []

# 逐一评估每个测试用例
for input_text, actual_output in tqdm(zip(inputs, actual_outputs), total=len(inputs), desc="Processing"):
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output
    )
    metric.measure(test_case)
    result = {
        "input": input_text,
        "actual_output": actual_output,
        "score": metric.score,
        "reason": metric.reason
    }
    results.append(result)

# 将结果写入JSON文件
dir_name, file_name = os.path.split(file_path2)
# 在文件名前添加 '_Score'
new_file_name = "_Score" + file_name
# 组合新的文件路径
store_path = os.path.join(dir_name, new_file_name)
# 输出新的文件路径
with open(store_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)

print("Results saved to ", store_path)
