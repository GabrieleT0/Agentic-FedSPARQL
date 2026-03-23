import dspy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from config import ENDPOINTS_FILE_PATH
import sparql_utils

# Agent 1: Discovery Agent
class Discovery(dspy.Module):
    def __init__(self, model_name: str = 'multi-qa-MiniLM-L6-cos-v1'):
        super().__init__()
        self.model = SentenceTransformer(model_name, device='cpu')
        self.index = {}
        with open(ENDPOINTS_FILE_PATH, 'r') as f:
            self.endpoints = json.load(f)
        self._build_index()

    def forward(self, question: str, discovery_attempts: int = 0) -> dspy.Prediction:
        k = 20 + discovery_attempts * 20
        question_embedding = self.model.encode(question)
        similarities = {url: cosine_similarity([embedding], [question_embedding])[0][0] 
                        for url, embedding in self.index.items()}
        sorted_endpoints = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Take a wider pool, then probe to filter to live/relevant ones
        pool = [url for url, _ in sorted_endpoints[:k * 2]]
        verified = []
        for url in pool:
            live_schema = sparql_utils.get_void_description(url)
            if live_schema["classes"] or live_schema["properties"]:
                # Re-score against live data
                live_text = ' '.join(live_schema["classes"]) + ' ' + ' '.join(live_schema["properties"])
                live_emb = self.model.encode(live_text)
                score = cosine_similarity([live_emb], [question_embedding])[0][0]
                verified.append((url, score))
            if len(verified) >= k:
                break
        
        verified.sort(key=lambda x: x[1], reverse=True)
        return dspy.Prediction(candidate_endpoints=[url for url, _ in verified[:k]])


    def _build_index(self):
        for el in self.endpoints:
            url = el['url']
            classes = el['classes']
            properties = el['properties']
            description = el.get('description', '')
            concatenated_description = ' '.join(classes) + ' ' + ' '.join(properties) + ' ' + description
            embedding = self.model.encode(concatenated_description)
            self.index[url] = embedding