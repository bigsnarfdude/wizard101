#!/usr/bin/env python3
"""
Simple FastAPI for Llama Guard refusal router.

Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from test_llama_guard import LlamaGuardRouter

app = FastAPI(
    title="Refusal Router API",
    description="Routes harmful content to appropriate refusal strategies using Llama Guard 3",
    version="0.1.0",
)

# Load model on startup
router = None


@app.on_event("startup")
async def load_model():
    global router
    router = LlamaGuardRouter()


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    safe: bool
    strategy: str
    categories: list[str]
    category_names: list[str]
    latency_ms: float


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """Classify text and return refusal strategy."""
    if router is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = router.classify(request.text)

    return ClassifyResponse(
        safe=result["safe"],
        strategy=result["strategy"],
        categories=result["categories"],
        category_names=result["category_names"],
        latency_ms=result["latency_ms"],
    )


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "model_loaded": router is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
