import dspy
from signatures import FilterSchema, IdentifyJoins
import sparql_utils
import json

# Agent 2: Schema Agent
class Schema(dspy.Module):
    def __init__(self):
        super().__init__()
        self.filter = dspy.ChainOfThought(FilterSchema)
        self.identify_joins = dspy.ChainOfThought(IdentifyJoins)

    def forward(self, question: str, candidate_endpoints: list) -> dspy.Prediction:
        void_descriptions = {}

        for endpoint in candidate_endpoints:
            void_descriptions[endpoint] = sparql_utils.get_void_description(endpoint)

        schema_summary = self.filter(question=question, void_descriptions=json.dumps(void_descriptions))
        identified_joins = self.identify_joins(question=question, schema_summary=schema_summary.schema_summary)
        
        return dspy.Prediction(schema_summary=schema_summary.schema_summary, join_candidates=identified_joins.join_candidates)