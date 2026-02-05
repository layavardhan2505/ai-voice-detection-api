# AI Voice Detection API

This project is a FastAPI-based REST API that detects whether a Base64-encoded MP3 voice is AI-generated or Human.

## Features
- Secure API with API key authentication
- Accepts Base64-encoded MP3 audio
- Supports Indian languages
- Extracts audio features using Librosa
- Returns classification with confidence and explanation

## How to Run

pip install -r requirements.txt
python -m uvicorn main:app --reload

## API Endpoint

POST /api/voice-detection

Header:
x-api-key: sk_test_123456789
