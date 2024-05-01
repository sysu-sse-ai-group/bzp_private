from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
import openai
import nest_asyncio

nest_asyncio.apply()
from llama_index.core import Settings, SummaryIndex, load_index_from_storage, StorageContext, Settings, \
    SimpleDirectoryReader, VectorStoreIndex, DocumentSummaryIndex, get_response_synthesizer
# openai.api_base = "https://api.chatanywhere.com.cn"
# openai.api_key = 'sk-BRIGHrNUlXTmWZxfUn7aOBfMlLh7LicIpRohEyMgDJYfo7q4'

# llm = OpenAI("gpt-3.5-turbo")
llm = Settings.llm = Ollama(base_url="http://localhost:11434",model="gemma")
evaluator = CorrectnessEvaluator(llm=llm)

query = (
    "Can you explain the theory of relativity proposed by Albert Einstein in"
    " detail?"
)

reference = """
Certainly! Albert Einstein's theory of relativity consists of two main components: special relativity and general relativity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for all non-accelerating observers and that the speed of light in a vacuum is a constant, regardless of the motion of the source or observer. It also gave rise to the famous equation E=mc², which relates energy (E) and mass (m).

General relativity, published in 1915, extended these ideas to include the effects of gravity. According to general relativity, gravity is not a force between masses, as described by Newton's theory of gravity, but rather the result of the warping of space and time by mass and energy. Massive objects, such as planets and stars, cause a curvature in spacetime, and smaller objects follow curved paths in response to this curvature. This concept is often illustrated using the analogy of a heavy ball placed on a rubber sheet, causing it to create a depression that other objects (representing smaller masses) naturally move towards.

In essence, general relativity provided a new understanding of gravity, explaining phenomena like the bending of light by gravity (gravitational lensing) and the precession of the orbit of Mercury. It has been confirmed through numerous experiments and observations and has become a fundamental theory in modern physics.
"""

response = """
fuck！I am a queencard
"""

result = evaluator.evaluate(
    query=query,
    response=response,
    reference=reference,
)

print(result)
print(result.score)