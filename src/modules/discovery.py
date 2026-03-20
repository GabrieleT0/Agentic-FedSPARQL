import dspy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from config import ENDPOINTS_FILE_PATH

# Agent 1: Discovery Agent
class Discovery(dspy.Module):
    def __init__(self, model_name: str = 'multi-qa-MiniLM-L6-cos-v1'):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.index = {}
        with open(ENDPOINTS_FILE_PATH, 'r') as f:
            self.endpoints = json.load(f)
        self._build_index()

    def forward(self, question: str, discovery_attempts: int = 0) -> dspy.Prediction:
        k = 5 + discovery_attempts * 5
        question_embedding = self.model.encode(question)
        similarities = {url: cosine_similarity([embedding], [question_embedding])[0][0] for url, embedding in self.index.items()}
        sorted_endpoints = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        top_k_endpoints = sorted_endpoints[:k]

        return dspy.Prediction(candidate_endpoints=[url for url, _ in top_k_endpoints])

    def _build_index(self):
        for el in self.endpoints:
            url = el['url']
            classes = el['classes']
            properties = el['properties']
            description = el.get('description', '')
            concatenated_description = ' '.join(classes) + ' ' + ' '.join(properties) + ' ' + description
            embedding = self.model.encode(concatenated_description)
            self.index[url] = embedding