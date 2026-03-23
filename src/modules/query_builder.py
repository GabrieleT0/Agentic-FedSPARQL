import dspy
from signatures import QueryBuilder
import ir_validator
import json

# Agent 3: Query Builder Agent
class QueryBuilderAgent(dspy.Module):
    def __init__(self, max_retries: int = 3):
        super().__init__()
        self.query_builder = dspy.ChainOfThought(QueryBuilder)
        self.max_retries = max_retries


    def forward(self, question: str, schema_summary: str, join_candidates: str, previous_error: str) -> dspy.Prediction:
        
        previous_error = previous_error if previous_error else "none"
        for attempt in range(self.max_retries):
            query_plan = self.query_builder(
                question=question,
                schema_summary=schema_summary,
                join_candidates=join_candidates,
                previous_error=previous_error
            )
            try:
                raw = query_plan.query_plan
                if isinstance(raw, dict):
                    json_ir = raw
                else:
                    # Strip markdown code fences if present
                    raw = raw.strip()
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    json_ir = json.loads(raw)
                print(json_ir)
            except (json.JSONDecodeError, IndexError) as e:
                print(f"JSON decoding error: {e}")
                previous_error = "Invalid JSON format in the generated query plan."
                continue
            is_valid, error_message = ir_validator.validate_ir(json_ir, json.loads(schema_summary))
            if is_valid:
                compiled_query = ir_validator.compile_ir_to_sparql(json_ir)
                return dspy.Prediction(sparql_query=compiled_query, json_ir=json_ir, success=True)
            previous_error = error_message
        return dspy.Prediction(sparql_query=None, json_ir=None, success=False, error_type=previous_error)
                