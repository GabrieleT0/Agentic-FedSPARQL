from pipeline import FederatedSPARQLPipeline
from config import MODE, BENCHMARK_DATA_PATH
from evaluate import evaluate

questions = [
    "Find the first names of the faculty members who are playing Canoeing or Kayaking.",
]

pipeline = FederatedSPARQLPipeline()

if MODE == "single":
    for question in questions:
        print(f"Question: {question}")
        
        result = pipeline(question=question)
        if result.success:
            print("SPARQL Query:")
            print(result.sparql_query)
            print("Query Results:")
            print(result.query_results)
        else:
            print("Failed to retrieve results.")
            print(result.error_type)
        print("\n" + "="*50 + "\n")
elif MODE == "benchmark":
    evaluate(pipeline, BENCHMARK_DATA_PATH, mode='full')
