import dspy
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
import mlflow
load_dotenv()

DEFAULT_LLM_MODEL = "lightning-ai/gemma-4-31B-it"


class DelayedLM(dspy.LM):
    """Add lightweight request spacing while retaining DSPy/LiteLLM retries."""

    def __init__(self, *args, request_delay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_delay = request_delay

    def forward(self, *args, **kwargs):
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        return super().forward(*args, **kwargs)

ollama_model = os.getenv("OLLAMA_MODEL")
llm_model = os.getenv("LLM_MODEL")

if not llm_model and ollama_model:
    llm_model = (
        ollama_model
        if ollama_model.startswith(("ollama/", "ollama_chat/"))
        else f"ollama_chat/{ollama_model}"
    )

llm_model = llm_model or DEFAULT_LLM_MODEL
lightning_request_delay = float(os.getenv("LIGHTNING_REQUEST_DELAY_SECONDS", "2"))
lightning_lm_retries = int(os.getenv("LIGHTNING_LM_RETRIES", "5"))

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("DSPy")
# mlflow.dspy.autolog()


dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

if llm_model.startswith(("ollama/", "ollama_chat/")):
    lm = dspy.LM(
        model=llm_model,
        api_base=os.getenv("OLLAMA_API_BASE", "http://host.docker.internal:11434"),
    )
elif "deepseek" in llm_model:
    lm = dspy.LM(model=f"openai/{llm_model}",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        )
elif "azure" in llm_model:  # openai / azure
    lm = dspy.LM(model=llm_model,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g. https://<your-resource>.openai.azure.com/
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
elif "lightning" in llm_model or 'google' in llm_model:
    lm = DelayedLM(model=f"openai/{llm_model}",
        api_key=os.getenv("LIGHTNING_API_KEY"),
        api_base=os.getenv("LIGHTNING_API_ENDPOINT"),
        request_delay=lightning_request_delay,
        num_retries=lightning_lm_retries,
        )

# elif "gpt" in llm_model:
#     lm = dspy.LM(model=llm_model,
#         api_key=os.getenv("OPENAI_API_KEY"),
#         api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
#         )
else:
    lm = dspy.LM(model=llm_model)

dspy.configure(lm=lm)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "9")) # Maximum number of retries for the entire pipeline
DISCOVERY_RETRY_LIMIT = int(os.getenv("DISCOVERY_RETRY_LIMIT", "6")) # Maximum retries inside Discovery per evaluated batch
xQUERY_BUILDER_RETRIES = int(os.getenv("QUERY_BUILDER_RETRIES", "6")) # Maximum retries inside QueryBuilderAgent per pipeline attempt
SCHEMA_SUMMARY_RETRY_LIMIT = int(os.getenv("SCHEMA_SUMMARY_RETRY_LIMIT", "9")) # Maximum retries for schema summary generation
ENDPOINTS_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/SPIDER4FedSPARQL/class-sharding/endpoints_metadata.json'))
FEDERATED_SPARQL_ENDPOINT = "http://host.docker.internal:3030/federated/sparql"
EXCLUDED_AGENT = os.getenv("EXCLUDED_AGENT", "")
BENCHMARK_DATA_PATH = os.path.abspath(os.getenv(
    "BENCHMARK_DATA_PATH",
    os.path.join(os.path.dirname(__file__), '../data/SPIDER4FedSPARQL/class-sharding/SPIDER4FedSPARQL_benchmark.json'),
))
BENCHMARK_RESULT_PATH = os.path.abspath(os.getenv(
    "BENCHMARK_RESULT_PATH",
    os.path.join(os.path.dirname(__file__), f'../data/benchmark_results/{llm_model}/benchmark_result_{MAX_RETRIES}_retries.json'),
))
os.makedirs(os.path.dirname(BENCHMARK_RESULT_PATH), exist_ok=True)
MODE = os.getenv("MODE", "benchmark") # single or benchmark

if MODE == "baseline":
    BENCHMARK_RESULT_PATH = os.path.abspath(os.getenv(
        "BENCHMARK_RESULT_PATH",
        os.path.join(os.path.dirname(__file__), f'../data/benchmark_results/{llm_model}/baseline_benchmark_result.json'),
    ))

if MODE == "ablation":
    BENCHMARK_RESULT_PATH = os.path.abspath(os.getenv(
        "BENCHMARK_RESULT_PATH",
        os.path.join(os.path.dirname(__file__), f'../data/benchmark_results/{llm_model}/ablation_{EXCLUDED_AGENT}_benchmark_result.json'),
    ))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20")) # Number of endpoints to evaluate in each batch during discovery for the LLM agent

def get_config_snapshot() -> dict:
    return {
        "llm_model": llm_model,
        "max_retries": MAX_RETRIES,
        "discovery_retry_limit": DISCOVERY_RETRY_LIMIT,
        "query_builder_retries": QUERY_BUILDER_RETRIES,
        "schema_summary_retry_limit": SCHEMA_SUMMARY_RETRY_LIMIT,
        "batch_size": BATCH_SIZE,
        "lightning_request_delay_seconds": lightning_request_delay,
        "lightning_lm_retries": lightning_lm_retries,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
