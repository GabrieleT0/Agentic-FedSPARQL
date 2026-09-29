import dspy
import json

from config import QUERY_BUILDER_RETRIES
from modules.discovery import Discovery
from modules.query_builder import QueryBuilderAgent
from modules.schema import Schema
from modules.validator import Validator
from modules.zero_shot import ZeroShotQuery
from baseline_pipeline import _get_void_description_cached
import sparql_utils


ABLATION_AGENTS = {"discovery", "schema", "query_builder", "validator"}

class AblationPipeline(dspy.Module):
    def __init__(self, excluded_agent: str, top_k: int = 60):
        super().__init__()

        if excluded_agent not in ABLATION_AGENTS:
            raise ValueError(
                f"Unknown ablation agent {excluded_agent!r}. "
                f"Choose one of: {', '.join(sorted(ABLATION_AGENTS))}"
            )


        self.excluded_agent = excluded_agent
        self.discovery = Discovery(use_llm_selection=excluded_agent != "discovery")
        self.schema = Schema() if excluded_agent != "schema" else None
        self.query_builder = (
            QueryBuilderAgent(QUERY_BUILDER_RETRIES) 
            if excluded_agent != "query_builder" else None )
        self.zero_shot = ZeroShotQuery(schema_guided=True) if excluded_agent == "query_builder" else None
        self.validator = Validator() if excluded_agent != "validator" else None
        self.top_k = top_k

    def forward(self, question: str) -> dspy.Prediction:
        discovery = self.discovery(question=question)
        candidate_endpoints = (discovery.candidate_endpoints[: self.top_k])

        # Replace Schema with unfiltered endpoint descriptions
        if self.excluded_agent == "schema":
            descriptions = {
                endpoint: _get_void_description_cached(endpoint)
                for endpoint in candidate_endpoints
            }
            schema_summary = json.dumps(descriptions)
            join_candidates = "[]"
            schema_retries = 0
        else:
            schema = self.schema(
                question=question, 
                candidate_endpoints=candidate_endpoints
            )
            schema_summary = schema.schema_summary
            join_candidates = schema.join_candidates
            schema_retries = schema.schema_summary_retries

        # Generate SPARQL directly from the schema agent outputs.
        if self.excluded_agent == "query_builder":
            zero_shot_result = self.zero_shot(
                question=question,
                schema_summary=schema_summary,
                join_candidates=join_candidates,
            )
            return dspy.Prediction(
                candidate_endpoints=candidate_endpoints,
                sparql_query=zero_shot_result.sparql_query,
                query_results=zero_shot_result.sparql_query_results,
                error_type=zero_shot_result.error_type,
                refinement_attempts=0,
                discovery_internal_retries=discovery.internal_retries,
                query_builder_internal_retries=0,
                schema_summary_retries=schema_retries,
            )

        query = self.query_builder(
            question=question,
            schema_summary=schema_summary,
            join_candidates=join_candidates,
            previous_error=None,
        )

        if not query.success:
            return dspy.Prediction(
                candidate_endpoints=candidate_endpoints,
                sparql_query=None,
                query_results=None,
                error_type=query.error_type,
                refinement_attempts=0,
                discovery_internal_retries=discovery.internal_retries,
                query_builder_internal_retries=query.internal_retries,
                schema_summary_retries=schema_retries,
            )

        if self.excluded_agent == "validator":
            try:
                results = sparql_utils.execute_sparql_query(query.sparql_query, candidate_endpoints)
                error = None
            except Exception as e:
                results = None
                error = str(e)

            return dspy.Prediction(
                candidate_endpoints=candidate_endpoints,
                sparql_query=query.sparql_query,
                json_ir=query.json_ir,
                query_results=results,
                error_type=error,
                refinement_attempts=0,
                discovery_internal_retries=discovery.internal_retries,
                query_builder_internal_retries=query.internal_retries,
                schema_summary_retries=schema_retries,
            )

        validation = self.validator(
            question=question,
            sparql_query=query.sparql_query,
            candidate_endpoints=candidate_endpoints,
            json_ir=query.json_ir,
        )
        
        return dspy.Prediction(
            candidate_endpoints=candidate_endpoints,
            sparql_query=query.sparql_query,
            json_ir=query.json_ir,
            query_results=validation.query_results,
            error_type=validation.diagnosis,
            refinement_attempts=0,
            discovery_internal_retries=discovery.internal_retries,
            query_builder_internal_retries=query.internal_retries,
            schema_summary_retries=schema_retries,
        )

