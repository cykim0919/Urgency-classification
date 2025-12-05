import sys
import os

# 현재 파일(api 폴더)를 Python 경로에 추가
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI
from emergency_inference import predict_emergency

app = FastAPI()

@app.get("/predict/")
def predict(text: str):
    result = predict_emergency(text)
    return {"input": text, "prediction": result}


@app.post("/predict/")
def predict(text: str):
    result = predict_emergency(text)
    return {"input": text, "prediction": result}
