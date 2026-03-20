import dspy
import sparql_utils
from signatures import DiagnoseEmptyResult
import json

# Agent 4: Validate SPARQL query and diagnose empty results
class Validator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.diagnose = dspy.Predict(DiagnoseEmptyResult)

    def forward(self, question: str, sparql_query: str, candidate_endpoints: list, json_ir: dict) -> dspy.Prediction:
        query_results = sparql_utils.execute_sparql_query(sparql_query)
        if len(query_results) > 0:
            return dspy.Prediction(is_valid=True, diagnosis=None, query_results=query_results)
        
        probe_results = {}

        for endpoint in candidate_endpoints:
            if endpoint not in probe_results:
                probe_results[endpoint] = {"classes": {}, "properties": {}}
                ir_endpoint = next((ep for ep in json_ir.get('endpoints', []) if ep['url'] == endpoint), None)
                if ir_endpoint:
                    patterns = ir_endpoint.get('patterns', [])
                    if patterns:
                        for pattern in patterns:
                            if pattern['predicate'] == "a" or pattern['predicate'] == "rdf:type":
                                probe_results[endpoint]["classes"][pattern['object']] = sparql_utils.probe_class(endpoint, pattern['object'])
                            else:
                                probe_results[endpoint]["properties"][pattern['predicate']] = len(sparql_utils.probe_property(endpoint, pattern['predicate'])) > 0
        diagnosis = self.diagnose(question=question, sparql_query=sparql_query, probe_results=json.dumps(probe_results)).diagnosis
        
        return dspy.Prediction(is_valid=False, diagnosis=diagnosis, query_results=None)
    