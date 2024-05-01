import json
import os

from deepeval.benchmarks.mmlu.task import MMLUTask
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU
from deepeval.benchmarks import BigBenchHard

benchmark = BigBenchHard(enable_cot=True)

class Phi3(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Phi3_mini"

model_path = "E:\model\Phi3_mini"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

Phi3 = Phi3(model=model, tokenizer=tokenizer)
benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
    n_shots=3
)

# Replace 'mistral_7b' with your own custom model
benchmark.evaluate(model=Phi3)
print(benchmark.overall_score)
print("Task-specific Scores: ", benchmark.task_scores)
file_path2 = ""
Socres = benchmark.overall_score
tasks = [MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY]
benchmark = MMLU(tasks=tasks)

dir_name, file_name = os.path.split(file_path2)
    # 在文件名前添加 '_Score'
new_file_name = "_ev_Hallu_" + file_name
    # 组合新的文件路径
store_path = os.path.join(dir_name, new_file_name)
    # 输出新的文件路径
with open(store_path, 'w', encoding='utf-8') as f:
        json.dump(Socres, f, indent=4)

print("Results saved to ", store_path)
