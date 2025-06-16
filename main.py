from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from services import RecommendationService
from qdrant import get_vectordb, init_vectordb
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

class RecomendationRequest(BaseModel):
    input_text: str

class RecommendationResponse(BaseModel):
    recommendations: List[dict]
    error: str = None

recommendation_service = RecommendationService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_vectordb()
    yield

app = FastAPI(lifespan=lifespan)

Origin = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=Origin,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecomendationRequest):
    vectordb = get_vectordb()
    try:
        print("Received input text:", request.input_text)
        res = recommendation_service.get_recommendations(request.input_text, vectordb)
        return JSONResponse(content=res, media_type="application/json")
    except Exception as e:
        return {"error": str(e)}