import json
import numpy as np


def calculate_average_from_json(file_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 转换字符串为浮点数
    numbers = [float(num.strip()) for num in data]

    # 使用numpy计算平均值
    average = np.mean(numbers)

    return average


# 示例使用
if __name__ == "__main__":
    file_path = "G:\data\_blockchain_solana\\bm25_answer\lm3_ev_S__bm25_answer_dol.json"
    average = calculate_average_from_json(file_path)
    print("Average value:", average)

