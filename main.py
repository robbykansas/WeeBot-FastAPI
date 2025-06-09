from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from services import RecommendationService
from qdrant import get_vectordb, init_vectordb
import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

class RecomendationRequest(BaseModel):
    input_text: str

recommendation_service = RecommendationService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_vectordb()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/recommendations", response_model=List[dict])
async def get_recommendations(request: RecomendationRequest):
    vectordb = get_vectordb()
    try:
        return recommendation_service.get_recommendations(request.input_text, vectordb)
    except Exception as e:
        return {"error": str(e)}