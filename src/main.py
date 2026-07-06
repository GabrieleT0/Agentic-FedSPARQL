from pipeline import FederatedSPARQLPipeline
from config import MODE, BENCHMARK_DATA_PATH, BENCHMARK_RESULT_PATH, get_config_snapshot
from evaluate import evaluate

questions = [
    "Please show the names and descriptions of aircrafts associated with airports that have a total number of passengers bigger than 10000000.",
]

if MODE == "single":
    pipeline = FederatedSPARQLPipeline()
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
    pipeline = FederatedSPARQLPipeline()
    evaluate(pipeline, BENCHMARK_DATA_PATH, mode='full', output_path=BENCHMARK_RESULT_PATH, config=get_config_snapshot())
elif MODE == "baseline":
    from baseline_pipeline import Baseline
    baseline_pipeline = Baseline()
    evaluate(baseline_pipeline, BENCHMARK_DATA_PATH, mode='full', output_path=BENCHMARK_RESULT_PATH, config=get_config_snapshot())
