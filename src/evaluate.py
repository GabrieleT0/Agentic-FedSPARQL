import json
import sparql_utils
from metrics import execution_accuracy, f1_score, discovery_accuracy

def load_benchmark(benchmark_path: str) -> list:
    """Load the benchmark dataset from a JSON file."""
    with open(benchmark_path, 'r') as f:
        return json.load(f)

def evaluate(pipeline, benchmark_path: str, mode: str = 'full'):
    examples = load_benchmark(benchmark_path)
    results = []

    for example in examples:
        question = example['question']
        gold_endpoints = list()
        endpoints_list = example['endpoints']
        for endpoint in endpoints_list:
            gold_endpoints.append(endpoint['url'])
        gold_answers = sparql_utils.execute_sparql_query(example['federated_sparql'])

        prediction = pipeline(question=question)
        predicted_endpoints = prediction.candidate_endpoints or []
        predicted_answers = prediction.query_results or []
        predicted_sparql = prediction.sparql_query

        discovery_acc = discovery_accuracy(predicted_endpoints, gold_endpoints)
        exec_acc = execution_accuracy(predicted_answers, gold_answers)
        f1 = f1_score(predicted_answers, gold_answers)
        results.append({
            "question": question,
            "predicted_endpoints": predicted_endpoints,
            "gold_endpoints": gold_endpoints,
            "predicted_sparql": predicted_sparql,
            "gold_answers": gold_answers,
            "predicted_answers": predicted_answers,
            "discovery_accuracy": discovery_acc,
            "execution_accuracy": exec_acc,
            "f1_score": f1
        })
    
    print(f"Average Discovery Accuracy: {sum(r['discovery_accuracy'] for r in results) / len(results):.4f}")
    print(f"Average Accuracy: {sum(r['execution_accuracy'] for r in results) / len(results):.4f}")
    print(f"Average F1 Score: {sum(r['f1_score'] for r in results) / len(results):.4f}")

    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results