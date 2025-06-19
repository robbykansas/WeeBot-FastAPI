import os
from jina import JinaAIEmbedding
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings.embeddings import Embeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

vectordb: Qdrant | None = None

class LangchainJinaEmbeddings(Embeddings):
    def __init__(self, jina_embedder: JinaAIEmbedding):
        self.jina_embedder = jina_embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [item["embedding"] for item in self.jina_embedder.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.jina_embedder.embed_query(text)


# def is_sentence_transformer_model(path: str) -> bool:
#     required_files = ["config.json", "modules.json"]
#     return all(os.path.isfile(os.path.join(path, f)) for f in required_files)

# def load_embedding(local_embedding) -> SentenceTransformer:
#     if is_sentence_transformer_model(local_embedding):
#         print(f"Model found locally at: {local_embedding}")
#         model = SentenceTransformer(local_embedding, trust_remote_code=True)
#     else:
#         print(f"Model not found in {local_embedding}. Downloading from Hugging Face...")
#         model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
#         model.save(local_embedding)

#     return local_embedding

async def init_vectordb():
    global vectordb
    if vectordb is not None:
        return vectordb

    collection_name = "anilist-2025-qdrant"
    qdrant = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY')
    )

    # loc="./hf_embed"
    # os.makedirs(loc, exist_ok=True)
    jina_api_key = "jina_d0c55237ee3d48e4971f01af99a582b3293BUJC1aiMpOz46tlPs3InYUAij"
    jina_model = "jina-embeddings-v3"

    jina_embedder = JinaAIEmbedding(api_key=jina_api_key, model=jina_model)
    embedding_function = LangchainJinaEmbeddings(jina_embedder)

    # embedding_function = HuggingFaceEmbeddings(
    #     model_name=load_embedding(loc),
    #     model_kwargs={"trust_remote_code": True},
    # )

    vectordb = Qdrant(
        client=qdrant,
        collection_name=collection_name,
        embeddings=embedding_function
    )

def get_vectordb() -> Qdrant:
    if vectordb is None:
        init_vectordb()
    return vectordb