import dspy
import os

lm = dspy.LM(model="openai/gpt-oss-20b",
    api_key=os.getenv("LIGHTNING_API_KEY"),
    api_base="https://lightning.ai/api/v1/")
dspy.configure(lm=lm)

MAX_RETRIES = 5 # Maximum number of retries for the entire pipeline
MAX_DISCOVERY_ATTEMPTS = 5 # Maximum number of attempts for the discovery phase
ENDPOINTS_FILE_PATH = '../data/endpoints_metadata.json'
FEDERATED_SPARQL_ENDPOINT = "http://host.docker.internal:3030/federated/sparql"
BENCHMARK_DATA_PATH = '../data/fkgqa_benchmark/'