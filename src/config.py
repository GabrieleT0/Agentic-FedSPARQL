import dspy
import os
from dotenv import load_dotenv
import mlflow
load_dotenv()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog()

lm = dspy.LM(model="openai/gpt-oss-120b",
    api_key=os.getenv("LIGHTNING_API_KEY"),
    api_base="https://lightning.ai/api/v1/",
    temperature=0.0)
dspy.configure(lm=lm)

MAX_RETRIES = 10 # Maximum number of retries for the entire pipeline
MAX_DISCOVERY_ATTEMPTS = 10 # Maximum number of attempts for the discovery phase
ENDPOINTS_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/SPIDER4FedSPARQL/endpoints_metadata.json'))
FEDERATED_SPARQL_ENDPOINT = "http://host.docker.internal:3030/federated/sparql"
BENCHMARK_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/SPIDER4FedSPARQL/SPIDER4FedSPARQL_benchmark.json'))
MODE = 'single'
BATCH_SIZE = 450 # Number of endpoints to evaluate in each batch during discovery for the LLM agent