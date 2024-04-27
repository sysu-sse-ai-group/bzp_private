from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SummaryIndex, load_index_from_storage, StorageContext, Settings, \
    SimpleDirectoryReader, VectorStoreIndex, DocumentSummaryIndex, get_response_synthesizer
from tqdm import tqdm
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
import datetime
import time
import chromadb
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers import SummaryIndexLLMRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core import QueryBundle
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.tools import RetrieverTool
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.postprocessor import SentenceTransformerRerank
import logging
import sys, os, json
from llama_index.core import Document

root_directory = 'G:\data'


def find_raw_query_files(root_dir):
    raw_query_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == '_raw_query.json':
                raw_query_files.append(os.path.join(dirpath, filename))
    return raw_query_files


def jsonfile_to_list(file_path):
    inputs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        # 仅读取前50个元素或者全部元素（如果少于50个）
        for item in data[:30]:
            inputs.append(item['query'])  # 提取'query'的值并添加到列表中
    return inputs


def setup_embedding_llm(embed_model, llm_model):
    ollama_embedding = OllamaEmbedding(
        model_name=llm_model,
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
        embed_batch_size=256,
    )
    llm = Settings.llm = Ollama(base_url="http://localhost:11434", model=llm_model, request_timeout=100)
    Settings.embed_model = ollama_embedding
    return ollama_embedding, llm


def load_documents(dir_path):
    print("本章sources:", dir_path)
    documents = SimpleDirectoryReader(input_dir=dir_path).load_data()
    return documents


def prepare_nodes_index(documents):
    splitter = SentenceSplitter(chunk_size=512)
    t1 = time.time()
    nodes = splitter.get_nodes_from_documents(
        [Document(text=documents[0].get_content()[:1000000])]
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    t2 = time.time()
    print("Indexing time:", t2 - t1)
    return index, t2 - t1


def query_engine_response(retriever, llm, query):
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
    )
    hyde = HyDEQueryTransform(include_original=True)
    hyde_query_engine = TransformQueryEngine(query_engine, hyde)
    response = hyde_query_engine.query(query)
    response = str(response)
    return response


def list_to_json_with_tags(output_list, filename, item):
    # Create a new list where each item is a dictionary with the key 'answer'
    formatted_data = [{'answer': item} for item in output_list]
    directory, filename = os.path.split(filename)
    # 将文件名中的'_raw_query.json'替换为'_raw_answer.json'
    answer_name = 'HyDE_answer_'+item+'.json'
    new_filename = filename.replace('_raw_query.json', answer_name)

    # 使用os.path.join()方法将目录路径和新的文件名拼接成新的路径
    store_path = os.path.join(directory, new_filename)
    print("储存到：", store_path)
    # Write the list of dictionaries to a JSON file
    with open(store_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)


def find_sources_directories(root):
    # 这个列表用来存储名为 "sources" 的目录的路径
    sources_directories = []
    # os.walk 通过在目录树中游走（从顶部向下或从底部向上），生成目录中的文件名。
    for dirpath, dirnames, filenames in os.walk(root):
        # 检查 'sources' 是否是 dirpath 的直接子目录
        if 'sources' in dirnames:
            # 构建到 'sources' 目录的完整路径
            full_path = os.path.join(dirpath, 'sources')
            sources_directories.append(full_path)

    return sources_directories

# def check_and_wait_until(target_hour=6):
#     """等到指定的小时数，如果已经超过，则立即执行；如果没有到，则等待。"""
#     print("waiting")
#     while True:
#         current_time = datetime.datetime.now()
#         if current_time.hour >= target_hour and current_time.minute < 1:
#             # 如果已经是早上6点，但还没过1分钟，就执行任务
#             break
#         elif current_time.hour < target_hour:
#             # 如果还没到6点，计算需要等待的时间
#             target_time = current_time.replace(hour=target_hour, minute=0, second=0, microsecond=0)
#             wait_seconds = (target_time - current_time).total_seconds()
#             print(f"Waiting for {wait_seconds} seconds until it's 6 AM.")
#             time.sleep(wait_seconds)
#             break
#         else:
#             # 如果已经超过6点1分钟，计算到第二天6点的等待时间
#             target_time = (current_time + datetime.timedelta(days=1)).replace(hour=target_hour, minute=0, second=0, microsecond=0)
#             wait_seconds = (target_time - current_time).total_seconds()
#             print(f"Waiting for {wait_seconds} seconds until the next 6 AM.")
#             time.sleep(wait_seconds)
#             break

# Example usage:
# check_and_wait_until()
sources_list = find_sources_directories(root_directory)
filelist = find_raw_query_files(root_directory)
print(filelist)
# [{'dol': 'dolphin-phi'}, {'ge_2': 'gemma:2b'}, {'ge_7': 'gemma:7b'},{'llama_13': 'llama2:13b'}, {'llama_2': 'llama2:7b'}, {'llava': 'llava'}, {'mistral': 'mistral'},{'openchat': 'openchat'}, {'qwen': 'qwen'}]
llm_model_list = [{'dol': 'dolphin-phi'}, {'ge_2': 'gemma:2b'},{'llama_2': 'llama2:7b'}, {'llava': 'llava'}, {'mistral': 'mistral'},{'openchat': 'openchat'}, {'qwen': 'qwen'}]
for llm_model in llm_model_list:
    item_name = next(iter(llm_model.keys()))
    model_name = llm_model[item_name]
    print("现在模型是：", llm_model)
    for source, file in zip(sources_list, filelist):
        documents = load_documents(source)
        ollama_embedding, llm = setup_embedding_llm('llama2:13b', model_name)
        index, indexing_time = prepare_nodes_index(documents)
        vector_retriever = VectorIndexRetriever(index, similarity_top_k=2)
        # bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=2)
        t1 = time.time()
        Inputs = jsonfile_to_list(file)
        Outputs = []
        print("问题长度：",len(Inputs))
        for query_str in tqdm(Inputs, desc="Processing Queries"):
            res = query_engine_response(vector_retriever, llm, query_str)
            Outputs.append(res)
        list_to_json_with_tags(Outputs, file, item_name)
        t2 = time.time()
        print("作答时间：",t2-t1)
        Outputs.clear()

# response = query_engine_response(bm25_retriever, llm, "What did Paul Graham do after RISD?")
# print("Query response:", response)
