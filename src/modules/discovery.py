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
from config import ENDPOINTS_FILE_PATH, BATCH_SIZE, DISCOVERY_RETRY_LIMIT
from signatures import EvaluateEndpoints, ExpandQuery

DENSE_WEIGHT_FIELDS = ("label", "classes", "properties", "description", "examples")
DEFAULT_DENSE_WEIGHTS = {
    "label": 0.10,
    "classes": 0.25,
    "properties": 0.30,
    "description": 0.20,
    "examples": 0.15,
}


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


class Discovery(dspy.Module):

    def __init__(
        self,
        model_name: str = 'nomic-ai/nomic-embed-text-v1.5',
        dense_weights: dict[str, float] | None = None,
        bm25_k1: float = 1.8,
        bm25_b: float = 0.75,
        bm25_epsilon: float = 0.25,
        batch_size: int = BATCH_SIZE,
        discovery_retry_limit: int = DISCOVERY_RETRY_LIMIT,
        rrf_k: int = 60,
    ):
        super().__init__()
        self.model = SentenceTransformer(model_name, device='cpu', trust_remote_code=True)
        self.expand_query = dspy.Predict(ExpandQuery)
        self.evaluate = dspy.ChainOfThought(EvaluateEndpoints)
        self.dense_weights = self._resolve_dense_weights(dense_weights)
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.bm25_epsilon = bm25_epsilon
        self.batch_size = batch_size
        self.discovery_retry_limit = discovery_retry_limit
        self.rrf_k = rrf_k
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
    def _resolve_dense_weights(dense_weights: dict[str, float] | None) -> dict[str, float]:
        weights = DEFAULT_DENSE_WEIGHTS.copy()
        if dense_weights:
            for field, value in dense_weights.items():
                if field not in DEFAULT_DENSE_WEIGHTS:
                    raise ValueError(f"Unknown dense weight field: {field}")
                weights[field] = float(value)

        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("Dense weights must sum to a positive value.")
        return {field: value / total for field, value in weights.items()}

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
            examples = el.get('examples', [])

            endpoint_label = url.rstrip('/').split('/')[-2].replace('__', ' ').replace('_', ' ')
            class_names = [self._local_name(c) for c in classes if 'rdf-syntax-ns' not in c]
            prop_names = [self._local_name(p) for p in properties if 'rdf-syntax-ns' not in p]
            # Extract readable tokens from example SPARQL queries
            example_text = ' '.join(
                self._local_name(token)
                for ex in examples
                for token in ex.replace('\n', ' ').split()
                if token.startswith('<') or '/' in token or '#' in token
            )

            self.static_descriptions[url] = {
                'endpoint': endpoint_label,
                'classes': class_names,
                'properties': prop_names,
                'description': description,
                'examples': examples,
            }

            self.index[url] = {
                'classes':     self.model.encode(' '.join(class_names)) if class_names else np.zeros(dim),
                'properties':  self.model.encode(' '.join(prop_names))  if prop_names  else np.zeros(dim),
                'description': self.model.encode(description)           if description  else np.zeros(dim),
                'label':       self.model.encode(endpoint_label),
                'examples':    self.model.encode(example_text)          if example_text else np.zeros(dim),
            }

    def _build_bm25(self):
        for url, desc in self.static_descriptions.items():
            example_tokens = ' '.join(
                self._local_name(token)
                for ex in desc.get('examples', [])
                for token in ex.replace('\n', ' ').split()
                if token.startswith('<') or '/' in token or '#' in token
            )
            text = (desc['endpoint'] + ' '
                    + ' '.join(desc['classes']) + ' '
                    + ' '.join(desc['properties']) + ' '
                    + desc['description'] + ' '
                    + example_tokens)
            self.bm25_docs.append(word_tokenize(text.lower()))
            self.bm25_urls.append(url)
        self.bm25_model = BM25Okapi(
            self.bm25_docs,
            k1=self.bm25_k1,
            b=self.bm25_b,
            epsilon=self.bm25_epsilon,
        )

    def _dense_scores(self, q_emb: np.ndarray) -> dict:
        """Weighted cosine similarity across per-field embeddings."""
        return {
            url: (self.dense_weights['label'] * _cosine(fields['label'], q_emb)
                + self.dense_weights['classes'] * _cosine(fields['classes'], q_emb)
                + self.dense_weights['properties'] * _cosine(fields['properties'], q_emb)
                + self.dense_weights['description'] * _cosine(fields['description'], q_emb)
                + self.dense_weights['examples'] * _cosine(fields['examples'], q_emb))
            for url, fields in self.index.items()
        }

    def _bm25_scores(self, q_tokens: list) -> dict:
        raw = self.bm25_model.get_scores(q_tokens)
        return {url: float(score) for url, score in zip(self.bm25_urls, raw)}

    @staticmethod
    def _normalize_relevant_endpoints(raw_endpoints, batch: list[str]) -> list[str]:
        if raw_endpoints is None:
            return []
        if not isinstance(raw_endpoints, list):
            raise ValueError(f"relevant_endpoints must be a list, got {type(raw_endpoints).__name__}")
        batch_set = set(batch)
        normalized = []
        seen = set()
        for endpoint in raw_endpoints:
            if not isinstance(endpoint, str):
                raise ValueError(f"relevant_endpoints entries must be strings, got {type(endpoint).__name__}")
            if endpoint not in batch_set:
                raise ValueError(f"relevant endpoint not present in current batch: {endpoint}")
            if endpoint not in seen:
                seen.add(endpoint)
                normalized.append(endpoint)
        return normalized

    @staticmethod
    def _normalize_is_sufficient(value) -> bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"is_sufficient must be a bool, got {type(value).__name__}")

    def forward(self, question: str, failed_endpoints: list = None) -> dspy.Prediction:
        failed = set(failed_endpoints or [])
        internal_retries = 0

        expanded = self.expand_query(question=question).expanded_question
        q_emb = self.model.encode(expanded)
        q_tokens = word_tokenize(expanded.lower())

        # Stage 1 — retrieval: dense + BM25, exclude already-failed endpoints
        dense_ranked = [url for url, _ in sorted(self._dense_scores(q_emb).items(),
                                                  key=lambda x: x[1], reverse=True)
                        if url not in failed]
        bm25_ranked  = [url for url, _ in sorted(self._bm25_scores(q_tokens).items(),
                                                  key=lambda x: x[1], reverse=True)
                        if url not in failed]

        # Stage 2 — fusion: Reciprocal Rank Fusion combines both ranked lists
        fused_scores = _rrf_fusion([dense_ranked, bm25_ranked], k=self.rrf_k)
        sorted_urls  = sorted(fused_scores, key=fused_scores.get, reverse=True)

        # Stage 3 — LLM reranking: batch evaluation with is_sufficient early stopping
        selected = []
        selected_descs = {}
        already_evaluated = set()
        max_batches = self.discovery_retry_limit

        for _ in range(max_batches):
            batch = [url for url in sorted_urls if url not in already_evaluated][:self.batch_size]
            if not batch:
                break
            already_evaluated.update(batch)
            batch_descriptions = json.dumps({url: self.static_descriptions[url] for url in batch})

            eval_result = None
            relevant = []
            is_sufficient = False
            for attempt in range(self.discovery_retry_limit):
                internal_retries += 1
                eval_result = self.evaluate(
                    question=question,
                    already_selected_descriptions=json.dumps(selected_descs),
                    candidate_descriptions=batch_descriptions,
                )
                try:
                    relevant = self._normalize_relevant_endpoints(eval_result.relevant_endpoints, batch)
                    is_sufficient = self._normalize_is_sufficient(eval_result.is_sufficient)
                    break
                except ValueError as e:
                    print(
                        f"Discovery agent attempt {attempt + 1}/{self.discovery_retry_limit}: "
                        f"invalid evaluation output ({e}), retrying..."
                    )
            else:
                print(
                    f"Discovery agent failed to produce valid evaluation output after "
                    f"{self.discovery_retry_limit} attempts; using no newly selected endpoints for this batch."
                )

            selected.extend(relevant)
            selected_descs.update({url: self.static_descriptions[url] for url in relevant})

            if is_sufficient:
                break

        # Fallback: if LLM filtered everything out, return top fusion results
        return dspy.Prediction(
            candidate_endpoints=selected if selected else sorted_urls[:self.batch_size],
            internal_retries=internal_retries,
        )
