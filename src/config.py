import dspy
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import mlflow
load_dotenv()

#llm_model = "deepseek/deepseek-chat"
llm_model = "azure/gpt-5-mini"


# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("DSPy")
# mlflow.dspy.autolog()


dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

if "deepseek" in llm_model:
    lm = dspy.LM(model=llm_model,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        )
elif "azure" in llm_model:  # openai / azure
    lm = dspy.LM(model=llm_model,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g. https://<your-resource>.openai.azure.com/
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )

dspy.configure(lm=lm)

MAX_RETRIES = 5 # Maximum number of retries for the entire pipeline
DISCOVERY_RETRY_LIMIT = 3 # Maximum retries inside Discovery per evaluated batch
QUERY_BUILDER_RETRIES = 3 # Maximum retries inside QueryBuilderAgent per pipeline attempt
SCHEMA_SUMMARY_RETRY_LIMIT = 3 # Maximum retries for schema summary generation
ENDPOINTS_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/SPIDER4FedSPARQL/class-sharding/endpoints_metadata.json'))
FEDERATED_SPARQL_ENDPOINT = "http://host.docker.internal:3030/federated/sparql"
BENCHMARK_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/SPIDER4FedSPARQL/class-sharding/SPIDER4FedSPARQL_benchmark.json'))
BENCHMARK_RESULT_PATH =  os.path.abspath(os.path.join(os.path.dirname(__file__), f'../data/benchmark_results/{llm_model}/benchmark_result_{MAX_RETRIES}_retries.json'))
os.makedirs(os.path.dirname(BENCHMARK_RESULT_PATH), exist_ok=True)
MODE = 'benchmark' # single or benchmark
BATCH_SIZE = 100 # Number of endpoints to evaluate in each batch during discovery for the LLM agent

def get_config_snapshot() -> dict:
    return {
        "llm_model": llm_model,
        "max_retries": MAX_RETRIES,
        "discovery_retry_limit": DISCOVERY_RETRY_LIMIT,
        "query_builder_retries": QUERY_BUILDER_RETRIES,
        "schema_summary_retry_limit": SCHEMA_SUMMARY_RETRY_LIMIT,
        "batch_size": BATCH_SIZE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
