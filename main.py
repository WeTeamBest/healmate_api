# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import predict

app = FastAPI(
    title="HealMate Emotion API",
    description="Prediksi emosi & healing score dari teks",
    version="1.0.0",
)


# ── Schema request & response ───────────────────────────────
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text_original : str
    text_english  : str  
    text_clean    : str
    emotion       : str
    confidence    : float
    all_emotions  : dict
    healing_score : float
    counselor_response: str
    activity_suggestions: list[str]


# ── Endpoint ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "HealMate Emotion API is running 🟢"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text tidak boleh kosong")
    
    result = predict(request.text)
    return result