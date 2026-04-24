import dspy
from signatures import FilterSchema, IdentifyJoins
import sparql_utils
import json
from functools import lru_cache
from config import SCHEMA_SUMMARY_RETRY_LIMIT

@lru_cache(maxsize=None)
def _get_void_description_cached(endpoint: str):
    return sparql_utils.get_void_description(endpoint)

# Agent 2: Schema Agent
class Schema(dspy.Module):
    def __init__(self):
        super().__init__()
        self.filter = dspy.ChainOfThought(FilterSchema)
        self.identify_joins = dspy.ChainOfThought(IdentifyJoins)

    def forward(self, question: str, candidate_endpoints: list) -> dspy.Prediction:
        void_descriptions = {}

        for endpoint in candidate_endpoints:
            void_descriptions[endpoint] = _get_void_description_cached(endpoint)

        void_descriptions_str = json.dumps(void_descriptions)
        schema_summary_str = None
        schema_summary_retries = 0
        for attempt in range(SCHEMA_SUMMARY_RETRY_LIMIT):
            result = self.filter(question=question, void_descriptions=void_descriptions_str)
            raw = result.schema_summary.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            try:
                json.JSONDecoder().raw_decode(raw.strip())
                schema_summary_str = result.schema_summary
                schema_summary_retries = attempt + 1
                break
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Schema agent attempt {attempt + 1}/{SCHEMA_SUMMARY_RETRY_LIMIT}: invalid JSON in schema_summary ({e}), retrying...")

        if schema_summary_str is None:
            schema_summary_retries = SCHEMA_SUMMARY_RETRY_LIMIT
            raise ValueError(f"Schema agent failed to produce valid JSON schema_summary after {SCHEMA_SUMMARY_RETRY_LIMIT} attempts.")

        identified_joins = self.identify_joins(question=question, schema_summary=schema_summary_str)

        return dspy.Prediction(schema_summary=schema_summary_str, join_candidates=identified_joins.join_candidates, schema_summary_retries=schema_summary_retries)