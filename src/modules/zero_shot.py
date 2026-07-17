import dspy
import json
import sparql_utils

from signatures import ZeroShotQuery as ZeroShotQuerySignature

class ZeroShotQuery(dspy.Module):
    def __init__(self):
        super().__init__()
        self.zero_shot_query = dspy.ChainOfThought(ZeroShotQuerySignature)

    def forward(self, question: str, void_descriptions: str) -> dspy.Prediction:
        error = None
        if isinstance(void_descriptions, dict):
            void_descriptions = json.dumps(void_descriptions)
        sparql_query = self.zero_shot_query(question=question, void_descriptions=void_descriptions).sparql_query
        sparql_query = sparql_utils.strip_sparql_code_fence(sparql_query)
        try:
            results = sparql_utils.execute_sparql_query(sparql_query)
        except Exception as e:
            print(f"Error occurred while executing SPARQL query: {e}")
            results = None
            error = str(e)
        return dspy.Prediction(sparql_query=sparql_query, sparql_query_results=results, error_type=error)
