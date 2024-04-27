import pandas as pd
import json, time
from llama_index.core.evaluation import DatasetGenerator, EvaluationResult
from llama_index.core.node_parser import SentenceSplitter
from llama_index.legacy import Response
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.core import Settings, SummaryIndex, load_index_from_storage, StorageContext, Settings, \
    SimpleDirectoryReader, VectorStoreIndex, DocumentSummaryIndex, get_response_synthesizer
from llama_index.core.evaluation import FaithfulnessEvaluator


t1 = time.time()
ollama_embedding = OllamaEmbedding(
    model_name="gemma:2b",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
    embed_batch_size=100
)

llm = Settings.llm = Ollama(base_url="http://localhost:11434",model="gemma:2b",)
Settings.embed_model = ollama_embedding
evaluator = FaithfulnessEvaluator(llm=llm)

documents = SimpleDirectoryReader(
    input_files=["G:\data\Chinese-medical-dialogue-data-master\Data_statistic\sample\\article.txt"]
).load_data()

t2 = time.time()
print("load_time", t2-t1)
splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter]
)

# 假设你有一个包含句子的列表
# 将每个句子和其对应的标签组成字典

# question_generator = DatasetGenerator.from_documents(
#     documents=documents,
#     llm=llm,
#     num_questions_per_chunk=5,
# )

t3 = time.time()
print("index_time", t3-t2)
question_generator = DatasetGenerator.from_documents(documents)
eval_questions = question_generator.generate_questions_from_nodes()

t4 = time.time()
print("generate_time", t4-t3)

labeled_sentences = [{"sentence": sentence, "label": "question"} for sentence in eval_questions]

# 将字典列表保存为 JSON 文件
with open('G:\index_storage\question_load\labeled_sentences_gemm.json', 'w') as f:
    json.dump(labeled_sentences, f, indent=4)
