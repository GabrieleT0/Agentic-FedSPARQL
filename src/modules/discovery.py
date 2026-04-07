import dspy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from config import ENDPOINTS_FILE_PATH, BATCH_SIZE
from signatures import EvaluateEndpoints

# Agent 1: Discovery Agent
class Discovery(dspy.Module):
    def __init__(self, model_name: str = 'nomic-ai/nomic-embed-text-v1.5'):
        super().__init__()
        self.model = SentenceTransformer(model_name, device='cpu', trust_remote_code=True)
        self.index = {}
        self.static_descriptions = {}
        self.evaluate = dspy.ChainOfThought(EvaluateEndpoints)
        with open(ENDPOINTS_FILE_PATH, 'r') as f:
            self.endpoints = json.load(f)
        self._build_index()

    def forward(self, question: str, discovery_attempts: int = 0, failed_endpoints: list = None) -> dspy.Prediction:
        failed = set(failed_endpoints or [])
        question_embedding = self.model.encode(question)
        similarities = {url: cosine_similarity([embedding], [question_embedding])[0][0]
                        for url, embedding in self.index.items()}
        sorted_endpoints = [url for url, _ in sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                            if url not in failed]

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

        # Fallback: if LLM filtered everything out, return the top batch
        return dspy.Prediction(candidate_endpoints=selected if selected else sorted_endpoints[:BATCH_SIZE])

    @staticmethod
    def _local_name(uri: str) -> str:
        """Extract a human-readable token from a URI (fragment or last path segment)."""
        name = uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]
        return name.replace('-', ' ').replace('_', ' ')

    def _build_index(self):
        for el in self.endpoints:
            url = el['url']
            classes = el.get('classes', [])
            properties = el.get('properties', [])
            description = el.get('description', '')

            # Derive a readable label from the endpoint URL path, e.g.
            # ".../activity_1__faculty/sparql" -> "activity 1 faculty"
            endpoint_label = url.rstrip('/').split('/')[-2] \
                .replace('__', ' ').replace('_', ' ')

            # Convert full URIs to local names; skip rdf:type noise
            class_names = [self._local_name(c) for c in classes
                           if 'rdf-syntax-ns' not in c]
            prop_names  = [self._local_name(p) for p in properties
                           if 'rdf-syntax-ns' not in p]

            self.static_descriptions[url] = {
                "endpoint": endpoint_label,
                "classes": class_names,
                "properties": prop_names,
                "description": description
            }
            concatenated = (endpoint_label + ' '
                            + ' '.join(class_names) + ' '
                            + ' '.join(prop_names)) + ' ' + description
            self.index[url] = self.model.encode(concatenated)
