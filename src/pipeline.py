from modules.discovery2 import Discovery2
from modules.schema import Schema
from modules.query_builder import QueryBuilderAgent
from modules.validator import Validator
import dspy
from config import MAX_RETRIES, QUERY_BUILDER_RETRIES, SCHEMA_SUMMARY_RETRY_LIMIT
class FederatedSPARQLPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.discovery = Discovery2()
        self.schema = Schema()
        self.query_builder = QueryBuilderAgent(QUERY_BUILDER_RETRIES)
        self.validator = Validator()

    def forward(self, question: str) -> dspy.Prediction:
        discovery_internal_retries = 0
        refinement_attempts = 0
        query_builder_internal_retries = 0
        schema_summary_retries = 0
        retry_from = "discovery"
        error = None
        failed_endpoints = set()
        while refinement_attempts < MAX_RETRIES:

            if retry_from == "discovery":
                discovery_result = self.discovery(
                    question=question,
                    failed_endpoints=list(failed_endpoints),
                )
                candidate_endpoints = discovery_result.candidate_endpoints
                discovery_internal_retries += discovery_result.internal_retries
                retry_from = "schema"

            if retry_from == 'schema':
                try:
                    schema_result = self.schema(question=question, candidate_endpoints=candidate_endpoints)
                    schema_summary_retries += schema_result.schema_summary_retries
                    retry_from = "query_builder"
                except ValueError as e:
                    error = str(e)
                    print(f"Schema agent failed: {error}. Retrying from discovery.")
                    schema_summary_retries += SCHEMA_SUMMARY_RETRY_LIMIT
                    retry_from = 'discovery'
                    refinement_attempts += 1
                    continue

            if retry_from == 'query_builder':
                query_builder_result = self.query_builder(question=question, schema_summary=schema_result.schema_summary, join_candidates=schema_result.join_candidates, previous_error=error)
                query_builder_internal_retries += query_builder_result.internal_retries

            if not query_builder_result.success:
                error = query_builder_result.error_type
                if error == "IR must contain at least one endpoint.":
                    retry_from = 'discovery'
                    refinement_attempts += 1
                else:
                    refinement_attempts += 1
                    retry_from = 'query_builder'
                continue

            validator_result = self.validator(question=question, sparql_query=query_builder_result.sparql_query, candidate_endpoints=candidate_endpoints, json_ir=query_builder_result.json_ir)
            if validator_result.is_valid:
                return dspy.Prediction(success=True, query_results=validator_result.query_results, sparql_query=query_builder_result.sparql_query, json_ir=query_builder_result.json_ir, error_type=None, candidate_endpoints=candidate_endpoints, refinement_attempts=refinement_attempts, discovery_internal_retries=discovery_internal_retries, query_builder_internal_retries=query_builder_internal_retries, schema_summary_retries=schema_summary_retries)
            else:
                error = validator_result.diagnosis
                wrong_endpoints = validator_result.wrong_endpoints
                if error == "wrong_endpoints":
                    retry_from = 'discovery'
                    print(f"Wrong endpoints identified: {wrong_endpoints}")
                    failed_endpoints.update(wrong_endpoints)
                    refinement_attempts += 1
                elif error == "wrong_schema":
                    refinement_attempts += 1
                    retry_from = 'schema'
                elif error == "wrong_query" or (error and error.startswith("SPARQL execution error")):
                    refinement_attempts += 1
                    retry_from = 'query_builder'
            print(f"Refinement attempt {refinement_attempts} failed with error: {error}")
            print("Generated SPARQL Query Results:")
            print(validator_result.query_results)
        return dspy.Prediction(success=False, query_results=None, sparql_query=None, json_ir=None, error_type=error, candidate_endpoints=candidate_endpoints, refinement_attempts=refinement_attempts, discovery_internal_retries=discovery_internal_retries, query_builder_internal_retries=query_builder_internal_retries, schema_summary_retries=schema_summary_retries)
