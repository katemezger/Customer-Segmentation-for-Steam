import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline import SteamSegmentationPipeline, PIPELINE_PATH

# module-level so the loaded model is shared across all requests
_pipeline: Optional[SteamSegmentationPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once at startup — load the model into memory so each request
    # doesn't pay the cost of deserializing the .pkl file
    global _pipeline
    if os.path.exists(PIPELINE_PATH):
        _pipeline = SteamSegmentationPipeline.load_pipeline(PIPELINE_PATH)
        print(f"Model loaded from {PIPELINE_PATH}")
    else:
        print(f"Warning: {PIPELINE_PATH} not found. Train the pipeline first.")
    yield


app = FastAPI(
    title="Steam Player Segmentation Engine",
    description="Real-time K-Means cluster inference for Steam user personas.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    # ge=0 means Pydantic rejects the request before it ever reaches the model
    total_play_hours: float = Field(..., ge=0, description="Sum of all play hours across games")
    average_play_hours: float = Field(..., ge=0, description="Mean play hours per game played")
    max_play_hours: float = Field(..., ge=0, description="Hours in single most-played game")
    unique_games_played: int = Field(..., ge=0, description="Count of games with any playtime")
    games_purchased: int = Field(..., ge=0, description="Total games purchased")
    play_to_purchase_ratio: float = Field(..., ge=0, description="unique_games_played / games_purchased")
    game_specific_hours: dict[str, float] = Field(
        default_factory=dict,
        description="Hours per named game title (zero-filled for unknown titles)",
    )


class PredictResponse(BaseModel):
    assigned_cluster: int
    assigned_persona: str
    marketing_action_item: str


@app.get("/", summary="Health check")
def health_check():
    return {"status": "healthy", "model_loaded": _pipeline is not None}


@app.post("/predict", response_model=PredictResponse, summary="Classify a Steam user into a persona")
def predict(request: PredictRequest):
    if _pipeline is None:
        # 503 = service exists but isn't ready, not a client error
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Run: python train.py to generate {PIPELINE_PATH}",
        )
    result = _pipeline.predict(request.model_dump())
    return PredictResponse(**result)
