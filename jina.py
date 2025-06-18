import requests
from typing import List

class JinaAIEmbedding:
    def __init__(self, api_key: str, model: str = "jina-embeddings-v3"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.jina.ai/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "task": "retrieval.query",
            "input": texts
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["data"]  # list of {"embedding": [...]}

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]["embedding"]