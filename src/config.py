import dspy
import os
from dotenv import load_dotenv
import mlflow
load_dotenv()

llm_model = "gpt-5-nano-2025-08-07"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog()

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
lm = dspy.LM(model=llm_model,
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=1)
dspy.configure(lm=lm)

MAX_RETRIES = 10 # Maximum number of retries for the entire pipeline
MAX_DISCOVERY_ATTEMPTS = 20 # Maximum number of attempts for the discovery phase
ENDPOINTS_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/SPIDER4FedSPARQL/endpoints_metadata.json'))
FEDERATED_SPARQL_ENDPOINT = "http://host.docker.internal:3030/federated/sparql"
BENCHMARK_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/SPIDER4FedSPARQL/SPIDER4FedSPARQL_benchmark.json'))
BENCHMARK_RESULT_PATH =  os.path.abspath(os.path.join(os.path.dirname(__file__), f'../data/benchmark_results/{llm_model}/benchmark_result.json'))
os.makedirs(os.path.dirname(BENCHMARK_RESULT_PATH), exist_ok=True)
MODE = 'benchmark' # single or benchmark
BATCH_SIZE = 450 # Number of endpoints to evaluate in each batch during discovery for the LLM agent