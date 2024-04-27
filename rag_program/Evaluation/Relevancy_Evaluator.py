import logging
import sys
from llama_index.core import (
    TreeIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
    Response,
)
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import RelevancyEvaluator, EvaluationResult
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
from llama_index.llms.ollama import Ollama

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

ollama_embedding = OllamaEmbedding(
    model_name="gemma:2b",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
    embed_batch_size=100
)
# gpt-3 (davinci)
llm_model = 'gemma:2b'
Settings.embed_model = ollama_embedding
model = Settings.llm = Ollama(base_url="http://localhost:11434", model=llm_model)
evaluator = RelevancyEvaluator(llm=model)
documents = SimpleDirectoryReader("G:\data\_covid19_paper\sources").load_data()

# create vector index
splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter]
)

def display_eval_df(
    query: str, response: Response, eval_result: EvaluationResult
) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": response.source_nodes[0].node.text[:1000] + "...",
            "Evaluation Result": "Pass" if eval_result.passing else "Fail",
            "Reasoning": eval_result.feedback,
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    print(eval_df)

query_str = (
    "According to the article, what actions should be taken to prevent future coronavirus outbreaks?"
)
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query_str)
eval_result = evaluator.evaluate_response(
    query=query_str, response=response_vector
)

print(eval_result.score)