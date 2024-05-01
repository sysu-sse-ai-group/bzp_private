import json

# 文件路径
file_path = "G:\data\_covid19_paper\\rag_dataset.json"


# 加载JSON数据
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

# 提取 'query' 字段

def extract_queries(data):
    queries = []
    examples = data.get('examples', [])  # 安全地获取 'examples' 键的值，如果不存在则返回空列表
    for item in examples:
        query = item.get('query')  # 安全地获取每个字典中的 'query' 键的值
        if query:  # 确保 'query' 键存在且不为空
            queries.append(query)
    return queries

# 使用函数
data = load_json(file_path)
queries = extract_queries(data)

formatted_data = [{"query": item} for item in queries]

# 写入到JSON文件
with open('G:\data\_covid19_paper\\raw_query.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)