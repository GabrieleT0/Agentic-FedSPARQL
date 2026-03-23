from modules.discovery import Discovery
from modules.schema import Schema
from modules.query_builder import QueryBuilderAgent
from modules.validator import Validator
import dspy
from config import MAX_RETRIES, MAX_DISCOVERY_ATTEMPTS
class FederatedSPARQLPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.discovery = Discovery()
        self.schema = Schema()
        self.query_builder = QueryBuilderAgent()
        self.validator = Validator()

    def forward(self, question: str) -> dspy.Prediction:
        discovery_attempts = 0
        refinement_attempts = 0
        retry_from = "discovery"
        error = None
        while refinement_attempts <= MAX_RETRIES:

            if retry_from == "discovery":
                discovery_result = self.discovery(question=question, discovery_attempts=discovery_attempts)
                candidate_endpoints = discovery_result.candidate_endpoints
                retry_from = "schema"

            if retry_from == 'schema':
                schema_result = self.schema(question=question, candidate_endpoints=candidate_endpoints)
                retry_from = "query_builder"

            if retry_from == 'query_builder':
                query_builder_result = self.query_builder(question=question, schema_summary=schema_result.schema_summary, join_candidates=schema_result.join_candidates, previous_error=error)

            if not query_builder_result.success:
                error = query_builder_result.error_type
                if error == "IR must contain at least one endpoint.":
                    retry_from = 'discovery'
                    discovery_attempts += 1
                    refinement_attempts += 1
                else:
                    refinement_attempts += 1
                    retry_from = 'query_builder'
                continue

            validator_result = self.validator(question=question, sparql_query=query_builder_result.sparql_query, candidate_endpoints=candidate_endpoints, json_ir=query_builder_result.json_ir)
            if validator_result.is_valid:
                return dspy.Prediction(success=True, query_results=validator_result.query_results, sparql_query=query_builder_result.sparql_query, json_ir=query_builder_result.json_ir, error_type=None, candidate_endpoints=candidate_endpoints)
            else:
                error = validator_result.diagnosis
                if error == "wrong_endpoints":
                    retry_from = 'discovery'
                    discovery_attempts += 1
                    refinement_attempts += 1
                elif error == "wrong_schema":
                    refinement_attempts += 1
                    retry_from = 'schema'
            print(f"Refinement attempt {refinement_attempts} failed with error: {error}")
            print("Generated SPARQL Query Results:")
            print(validator_result.query_results)
        return dspy.Prediction(success=False, query_results=None, sparql_query=None, json_ir=None, error_type=error, candidate_endpoints=candidate_endpoints)