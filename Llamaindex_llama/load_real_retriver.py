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

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers import SummaryIndexLLMRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.tools import RetrieverTool

# load documents
class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj


ollama_embedding = OllamaEmbedding(
    model_name="llama2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
    embed_batch_size=100
)

documents = SimpleDirectoryReader("G:\data\Chinese-medical-dialogue-data-master\Data_statistic\Pediatric1").load_data()
llm = Settings.llm = Ollama(base_url="http://localhost:11434",model="llama2",)
Settings.embed_model = ollama_embedding
splitter = SentenceSplitter(chunk_size=1024)
print("1")
t1 = time.time()
nodes = splitter.get_nodes_from_documents(documents)

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)


index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)
index.storage_context.persist("G:\index_storage\\aaa1")
# storage_context = StorageContext.from_defaults(persist_dir="G:\index_storage\men")
# index = load_index_from_storage(storage_context)


t2 = time.time()
print("store",t2-t1)
vector_retriever = VectorIndexRetriever(index)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)


synthesizer = get_response_synthesizer(
    llm=llm,
    response_mode=ResponseMode.SIMPLE_SUMMARIZE,
    use_async=False,
    streaming=False,
)

query_engine = RAGQueryEngine(
    retriever= bm25_retriever, response_synthesizer=synthesizer,llm=llm
)
t3 = time.time()
print("retriever:",t3-t2)
response = query_engine.query("早泄怎么办,可以用中文回答吗")
print(response)