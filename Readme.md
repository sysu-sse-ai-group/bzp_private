# 汇报进度

## 项目概况

 对大模型高效推理做研究，对文本生成，代码生成等方面嵌入RAG（检索增强生成）等技术的微调，再看能否将nlp领域的思维链引入

## 进度更新

### 本周工作完成情况

- 阅读论文：KnowGPT: Black-Box Knowledge Injection for Large Language     Models
  了解了KnowGPT的黑盒知识注入框架

- 阅读论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks 
   了解到RAG是参数化知识＋非参数化知识的结合，所以即是检索不到有用的相关文档，也可以单凭参数化的知识(BART)，根据模型学到的知识回答出正确的答案；
   RAG的灵活性很高。要改变预训练的语言模型所知道的内容，需要用新的文档对整个模型进行再训练。通过 RAG，我们可以通过交换知识检索所用的文档来控制它所知道的内容。

- 寻找到初步的数据（包括论文，算法代码已经试验结果），并且进行了embedding，还进行了余弦相似度对比

## 问题

- 问题1：用哪些模型，只用一些nlp模型可以吗？（chatgpt，Llama，BERT，WenBERT）
- 问题2：在embedding部分要不要分析

## 下一步计划

- 寻找更多大语言模型的代表
- 寻找更有说服力的训练例子
- 了解评估体系

## 资源需求

- 一些大模型体量太大，训练需要较大的计算资源，希望得到学院的服务器利用权限

2024/2/8

- 制作了rag相关的ppt学习文件

2024/2/20

- 探索THUDM的chatglm模型
