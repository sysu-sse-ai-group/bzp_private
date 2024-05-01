import os
import actual_output
import expected_output
import retrieval_context

from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
os.environ['OPENAI_API_BASE'] = "https://api.chatanywhere.cn/v1"
os.environ['OPENAI_API_KEY'] = 'sk-BRIGHrNUlXTmWZxfUn7aOBfMlLh7LicIpRohEyMgDJYfo7q4'
# Replace this with the actual output from your LLM application
actual_output = actual_output

# Replace this with the expected output from your RAG generator
expected_output =expected_output

# Replace this with the actual retrieved context from your RAG pipeline
retrieval_context = retrieval_context

metric = ContextualPrecisionMetric(
    threshold=0.5,
    model="gpt-3.5-turbo",
    include_reason=True
)
test_case = LLMTestCase(
    input=input(),
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)

metric.measure(test_case)
store(metric)

