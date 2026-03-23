import dspy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from config import ENDPOINTS_FILE_PATH, BATCH_SIZE
from signatures import EvaluateEndpoints


# Agent 1: Discovery Agent
class Discovery(dspy.Module):
    def __init__(self, model_name: str = 'multi-qa-MiniLM-L6-cos-v1'):
        super().__init__()
        self.model = SentenceTransformer(model_name, device='cpu')
        self.index = {}
        self.static_descriptions = {}
        with open(ENDPOINTS_FILE_PATH, 'r') as f:
            self.endpoints = json.load(f)
        self._build_index()
        self.evaluate = dspy.ChainOfThought(EvaluateEndpoints)

    def forward(self, question: str, discovery_attempts: int = 0) -> dspy.Prediction:
        question_embedding = self.model.encode(question)
        similarities = {url: cosine_similarity([embedding], [question_embedding])[0][0]
                        for url, embedding in self.index.items()}
        sorted_endpoints = [url for url, _ in sorted(similarities.items(), key=lambda x: x[1], reverse=True)]

        already_evaluated = set()
        selected = []         
        selected_descs = {}    
        max_batches = 1 + discovery_attempts

        for _ in range(max_batches):
            batch = [url for url in sorted_endpoints if url not in already_evaluated][:BATCH_SIZE]
            if not batch:
                break
            already_evaluated.update(batch)

            batch_descriptions = {url: self.static_descriptions[url] for url in batch}
            eval_result = self.evaluate(
                question=question,
                already_selected_descriptions=json.dumps(selected_descs),
                candidate_descriptions=json.dumps(batch_descriptions),
            )

            relevant = [url for url in (eval_result.relevant_endpoints or []) if url in batch]
            selected.extend(relevant)
            selected_descs.update({url: self.static_descriptions[url] for url in relevant})

            if eval_result.is_sufficient:
                break
        print("Selected endpoints:", selected)
        # Fallback: if LLM filtered everything out, return the top batch
        return dspy.Prediction(candidate_endpoints=selected if selected else sorted_endpoints[:BATCH_SIZE])

    def _build_index(self):
        for el in self.endpoints:
            url = el['url']
            classes = el.get('classes', [])
            properties = el.get('properties', [])
            description = el.get('description', '')
            self.static_descriptions[url] = {
                "classes": classes,
                "properties": properties,
                "description": description,
            }
            concatenated = ' '.join(classes) + ' ' + ' '.join(properties) + ' ' + description
            self.index[url] = self.model.encode(concatenated)
