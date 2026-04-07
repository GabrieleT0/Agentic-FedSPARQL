import dspy
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from config import ENDPOINTS_FILE_PATH, BATCH_SIZE
from signatures import EvaluateEndpoints


def _cosine(a, b):
    return float(cosine_similarity([a], [b])[0][0])


def _rrf_fusion(rankings: list, k: int = 60) -> dict:
    """Reciprocal Rank Fusion over multiple ranked lists of URLs.
    Score = sum of 1 / (k + rank) across all lists. Rank-based, no normalization needed."""
    scores = {}
    for ranked in rankings:
        for rank, url in enumerate(ranked):
            scores[url] = scores.get(url, 0.0) + 1.0 / (k + rank + 1)
    return scores


class Discovery2(dspy.Module):

    def __init__(self, model_name: str = 'nomic-ai/nomic-embed-text-v1.5'):
        super().__init__()
        self.model = SentenceTransformer(model_name, device='cpu', trust_remote_code=True)
        self.evaluate = dspy.ChainOfThought(EvaluateEndpoints)
        self.index = {}
        self.static_descriptions = {}
        self.bm25_docs = []
        self.bm25_urls = []
        self.bm25_model = None

        with open(ENDPOINTS_FILE_PATH, 'r') as f:
            self.endpoints = json.load(f)

        self._build_index()
        self._build_bm25()

    @staticmethod
    def _local_name(uri: str) -> str:
        """Extract a human-readable token from a URI (fragment or last path segment)."""
        name = uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]
        return name.replace('-', ' ').replace('_', ' ')

    def _build_index(self):
        dim = self.model.get_sentence_embedding_dimension()
        for el in self.endpoints:
            url = el['url']
            classes = el.get('classes', [])
            properties = el.get('properties', [])
            description = el.get('description', '')

            endpoint_label = url.rstrip('/').split('/')[-2].replace('__', ' ').replace('_', ' ')
            class_names = [self._local_name(c) for c in classes if 'rdf-syntax-ns' not in c]
            prop_names = [self._local_name(p) for p in properties if 'rdf-syntax-ns' not in p]

            self.static_descriptions[url] = {
                'endpoint': endpoint_label,
                'classes': class_names,
                'properties': prop_names,
                'description': description,
            }

            self.index[url] = {
                'classes':     self.model.encode(' '.join(class_names)) if class_names else np.zeros(dim),
                'properties':  self.model.encode(' '.join(prop_names))  if prop_names  else np.zeros(dim),
                'description': self.model.encode(description)           if description  else np.zeros(dim),
                'label':       self.model.encode(endpoint_label),
            }

    def _build_bm25(self):
        for url, desc in self.static_descriptions.items():
            text = (desc['endpoint'] + ' '
                    + ' '.join(desc['classes']) + ' '
                    + ' '.join(desc['properties']) + ' '
                    + desc['description'])
            self.bm25_docs.append(word_tokenize(text.lower()))
            self.bm25_urls.append(url)
        self.bm25_model = BM25Okapi(self.bm25_docs)

    def _dense_scores(self, q_emb: np.ndarray) -> dict:
        """Weighted cosine similarity across per-field embeddings."""
        return {
            url: (0.10 * _cosine(fields['label'],       q_emb)
                + 0.30 * _cosine(fields['classes'],     q_emb)
                + 0.40 * _cosine(fields['properties'],  q_emb)
                + 0.20 * _cosine(fields['description'], q_emb))
            for url, fields in self.index.items()
        }

    def _bm25_scores(self, q_tokens: list) -> dict:
        raw = self.bm25_model.get_scores(q_tokens)
        return {url: float(score) for url, score in zip(self.bm25_urls, raw)}

    def forward(self, question: str, discovery_attempts: int = 0, failed_endpoints: list = None) -> dspy.Prediction:
        failed = set(failed_endpoints or [])

        q_emb = self.model.encode(question)
        q_tokens = word_tokenize(question.lower())

        # Stage 1 — retrieval: dense + BM25, exclude already-failed endpoints
        dense_ranked = [url for url, _ in sorted(self._dense_scores(q_emb).items(),
                                                  key=lambda x: x[1], reverse=True)
                        if url not in failed]
        bm25_ranked  = [url for url, _ in sorted(self._bm25_scores(q_tokens).items(),
                                                  key=lambda x: x[1], reverse=True)
                        if url not in failed]

        # Stage 2 — fusion: Reciprocal Rank Fusion combines both ranked lists
        fused_scores = _rrf_fusion([dense_ranked, bm25_ranked])
        sorted_urls  = sorted(fused_scores, key=fused_scores.get, reverse=True)

        # Stage 3 — LLM reranking: batch evaluation with is_sufficient early stopping
        selected = []
        selected_descs = {}
        already_evaluated = set()
        max_batches = 1 + discovery_attempts

        for _ in range(max_batches):
            batch = [url for url in sorted_urls if url not in already_evaluated][:BATCH_SIZE]
            if not batch:
                break
            already_evaluated.update(batch)

            eval_result = self.evaluate(
                question=question,
                already_selected_descriptions=json.dumps(selected_descs),
                candidate_descriptions=json.dumps({url: self.static_descriptions[url] for url in batch}),
            )

            relevant = [url for url in (eval_result.relevant_endpoints or []) if url in batch]
            selected.extend(relevant)
            selected_descs.update({url: self.static_descriptions[url] for url in relevant})

            if eval_result.is_sufficient:
                break

        # Fallback: if LLM filtered everything out, return top fusion results
        return dspy.Prediction(candidate_endpoints=selected if selected else sorted_urls[:BATCH_SIZE])
