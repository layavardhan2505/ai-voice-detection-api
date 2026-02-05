from fastapi.security.api_key import APIKeyHeader
from fastapi import Depends
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import os
import uuid

API_KEY = "sk_test_123456789"

app = FastAPI(title="AI Voice Detection API")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitch = librosa.yin(y=y, fmin=50, fmax=300)
    pitch_var = np.var(pitch)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_entropy = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
    return mfcc.var(), pitch_var, zcr, spectral_entropy

def classify(features):
    mfcc_var, pitch_var, zcr, entropy = features
    if pitch_var < 40 and mfcc_var < 120:

        return "AI_GENERATED", 0.9, "Unnatural pitch consistency and robotic speech patterns detected"
    else:
        return "HUMAN", 0.85, "Natural pitch variation and human speech irregularities detected"

@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None, alias="x-api-key")
):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail={
            "status": "error",
            "message": "Invalid API key or malformed request"
        })

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "message": "Only MP3 format is supported"
        })

    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "message": "Invalid Base64 audio"
        })

    temp_filename = f"temp_{uuid.uuid4().hex}.mp3"
    temp_path = os.path.join(os.getcwd(), temp_filename)

    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    try:
        y, sr = librosa.load(temp_path, sr=16000)
    except Exception:
        os.remove(temp_path)
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Invalid or corrupted MP3 audio"
            }
        )

    os.remove(temp_path)

    features = extract_features(y, sr)
    classification, confidence, explanation = classify(features)

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

