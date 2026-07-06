from modules.zero_shot import ZeroShotQuery
from modules.discovery2 import Discovery2
import dspy
from config import MAX_RETRIES, QUERY_BUILDER_RETRIES, SCHEMA_SUMMARY_RETRY_LIMIT
import sparql_utils
from functools import lru_cache

@lru_cache(maxsize=None)
def _get_void_description_cached(endpoint: str):
    return sparql_utils.get_void_description(endpoint)

class Baseline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.discovery = Discovery2()
        self.zero_shot_query = ZeroShotQuery()
    
    def forward(self, question: str) -> dspy.Prediction:
        void_descriptions = {}
        failed_endpoints = set()
        discovery_result = self.discovery(
            question=question,
            failed_endpoints=list(failed_endpoints),
        )
        candidate_endpoints = discovery_result.candidate_endpoints

        for endpoint in candidate_endpoints:
            void_descriptions[endpoint] = _get_void_description_cached(endpoint)

        zero_shot_result = self.zero_shot_query(question=question, void_descriptions=void_descriptions)
        sparql_query = zero_shot_result.sparql_query
        sparql_query_results = zero_shot_result.sparql_query_results
        error_type = zero_shot_result.error_type

        return dspy.Prediction(sparql_query=sparql_query, query_results=sparql_query_results, candidate_endpoints=candidate_endpoints, refinement_attempts=0, json_ir=None, error_type=error_type, discovery_internal_retries=0, query_builder_internal_retries=0, schema_summary_retries=0)