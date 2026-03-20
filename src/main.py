from pipeline import FederatedSPARQLPipeline
import config

questions = [
    "What are the names of the movies directed by Christopher Nolan and starring Leonardo DiCaprio?",
    "Which actors have won an Oscar for Best Actor in a Leading Role and have also starred in a movie directed by Quentin Tarantino?",
]

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