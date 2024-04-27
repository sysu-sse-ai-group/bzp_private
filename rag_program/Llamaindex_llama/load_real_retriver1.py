from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SummaryIndex, load_index_from_storage, StorageContext, Settings, \
    SimpleDirectoryReader, VectorStoreIndex, DocumentSummaryIndex, get_response_synthesizer

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
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
import sys
from llama_index.core import Document


from llama_index.core.tools import RetrieverTool
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# load documents
ollama_embedding = OllamaEmbedding(
    model_name="llama2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
    embed_batch_size=100
)
documents = SimpleDirectoryReader(
    input_dir= "G:\data\_covid19_paper\sources"
).load_data()
llm = Settings.llm = Ollama(base_url="http://localhost:11434",model="llama2",)
Settings.embed_model = ollama_embedding
splitter = SentenceSplitter(chunk_size=512)
t1 = time.time()
nodes = splitter.get_nodes_from_documents(
    [Document(text=documents[0].get_content()[:1000000])]
)
# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)
t2 = time.time()
print("index_time:",t2-t1)

vector_retriever = VectorIndexRetriever(index, similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=2)

# nodes = bm25_retriever.retrieve("What happened at Viaweb and Interleaf?")
# nodes1 = bm25_retriever.retrieve("author mentions a quick decision. What was the decision he plan to execute it?")

t3 = time.time()
print("bm25:",t3-t2)
# for node in nodes:
#     print(node)

print("-----------above_bm25_retriever_region---------------------")

query_engine = RetrieverQueryEngine.from_args(
    retriever=bm25_retriever,
    llm=llm,
)

response = query_engine.query(
    "According to the article, what actions should be taken to prevent future coronavirus outbreaks?"
)

print("response_answer:",response)

#
nodes2 = vector_retriever.retrieve("What happened at Viaweb and Interleaf?")
nodes3 = vector_retriever.retrieve("author mentions a quick decision. What was the decision he plan to execute it?")

t4 = time.time()
print("vector:",t4-t3)

print("-----------above_vector_retriever_region---------------------")

retriever_tools = [
    RetrieverTool.from_defaults(
        retriever=vector_retriever,
        description="Useful in most cases",
    ),
    RetrieverTool.from_defaults(
        retriever=bm25_retriever,
        description="Useful if searching about specific information",
    ),
]

retriever = RouterRetriever.from_defaults(
    retriever_tools=retriever_tools,
    llm=llm,
    select_multi=True,
)

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

index.as_retriever(similarity_top_k=2)

hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

reranker = SentenceTransformerRerank(top_n=10, model="G:\model\\bge_reranker_base")
t5 = time.time()
retrieved_nodes = hybrid_retriever.retrieve("What did Paul Graham do after RISD?")
retrieved_nodes1 = hybrid_retriever.retrieve("author mentions a quick decision. What was the decision he plan to execute it?")

t6 = time.time()
print("bm+vector:",t6-t5)

reranked_nodes = reranker.postprocess_nodes(
    retrieved_nodes,
    query_bundle=QueryBundle(
        "What did Paul Graham do after RISD?",
    ),
)

reranked_nodes1 = reranker.postprocess_nodes(
    retrieved_nodes1,
    query_bundle=QueryBundle(
        "author mentions a quick decision. What was the decision he plan to execute it?",
    ),
)

t7 = time.time()
print("rerank:",t7-t6)

print("Initial retrieval: ", len(retrieved_nodes), " nodes")
print("Re-ranked retrieval: ", len(reranked_nodes), " nodes")

for node in retrieved_nodes:
    print(node)
print("---------------------------------------")
print("---------------------------------------")
for node in reranked_nodes:
    print(node)

t3 = time.time()
print("retriever_time:  ", t3-t2)

query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
    llm=llm,
)

response = query_engine.query(
    "What did Paul Graham do after RISD?"
)

print("response_answer:",response)

print("")
t4 = time.time()
print("query_time:  ", t4-t3)

synthesizer = get_response_synthesizer(
    llm=llm,
    response_mode=ResponseMode.SIMPLE_SUMMARIZE,
    use_async=False,
    streaming=False,
)

query_engine = RAGQueryEngine(
    retriever= retriever, response_synthesizer=synthesizer,llm=llm
)
t3 = time.time()
print("retriever:",t3-t2)
response = query_engine.query("早泄怎么办,可以用中文回答吗")
print(response)