# 汇报进度

## 项目概况

 对大模型高效推理做研究，对文本生成，代码生成等方面嵌入RAG（检索增强生成）等技术的微调，再看能否将nlp领域的思维链引入

## 进度更新

####  已完成：
### 一、数据收集：

医患数据：https://github.com/Toyhom/Chinese-medical-dialogue-data/tree/master/Data_%E6%95%B0%E6%8D%AE （对话型：体量较大）<br />
保罗·格雷厄姆的文章：https://github.com/run-llama/llama-datasets/tree/main/llama_datasets/paul_graham_essay （文章型）

### 二、平台搭建
①：Ollama＋llamaindex（已测试3项大语言模型：llama2_7b, llava, dolphin-phi）<br />
②：chatglm＋langchain（只有测试chatglm的模型）<br />
③：（计划后期加入chatgpt3.5），但是还未找到合适购买的api<br />

### 三、数据load
①  暂时数据都统一采用txt格式：loader用llamaindex官方通用的SimpleDirectoryReader<br />
②  embedding模型都使用的是llama2_7b<br />
③  分片器splitter = SentenceSplitter(chunk_size=1024)<br />

### 四、RAG模块<br />
④  retriver检索器提升：
查询用的是BM25Retriever，查询引擎是CustomQueryEngine
```
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
synthesizer = get_response_synthesizer(response_mode="compact")
query_engine = RAGQueryEngine(
    retriever= bm25_retriever, response_synthesizer=synthesizer,llm=llm
)
```
<br />
② retriver检索器提升：在①的基础上再加上重排序
```
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")
```
<br />
③ 假设性问题法HyDE（仍然在实验中）
需要LLM生成假设性问题，比如 ChatGPT，在响应查询时创建一个理论文档，而不是使用查询及其计算出的向量直接在向量数据库中搜索。
然后搜索嵌入向量以找到匹配项。在这里，我们进行比较的是答案到答案的嵌入相似性搜索，而不是传统 RAG 检索方法中的查询到答案的嵌入相似性搜索。

####  未完成：
### 五、评估模块
想法1：UHGEval<br />
论文地址：https://arxiv.org/abs/2311.15296<br />
代码地址：https://github.com/IAAR-Shanghai/UHGEval<br />

想法2：HalluQA<br />
论文地址：https://arxiv.org/abs/2310.03368<br />
代码地址：https://github.com/OpenMOSS/HalluQA<br />

问题：上述两个用的都是普适的问题进行幻觉评测，不知道如何与本地知识库联系<br />

想法3：llamaindex官方DeepEval的RAG/LLM 评估器<br />
有自带的（Bias，Toxicity，Faithfulness，Contextual Recall）都可以视为泛幻觉领域<br />
```
from deepeval.integrations.llama_index import (
    DeepEvalAnswerRelevancyEvaluator,
    DeepEvalFaithfulnessEvaluator,
    DeepEvalContextualRelevancyEvaluator,
    DeepEvalSummarizationEvaluator,
    DeepEvalBiasEvaluator,
    DeepEvalToxicityEvaluator,
)
```

### 六、搭建网站<br />
可以自己载入4种格式（pdf，txt，csv，doc）的本地知识库，通过RAG来改善回答。


