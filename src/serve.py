from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Đọc tên bucket từ biến môi trường (được đặt trong systemd service)
GCS_BUCKET = os.environ["GCS_BUCKET"]
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

def download_model():
    """Tải file model.pkl từ GCS về máy khi server khởi động."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_KEY)
    blob.download_to_filename(MODEL_PATH)
    print("Model downloaded successfully!")

# Gọi hàm này khi module được import (chạy khi server khởi động)
download_model()
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server."""
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    """Endpoint suy luận."""
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    pred = model.predict([req.features])[0]
    
    labels = {0: "thấp", 1: "trung_bình", 2: "cao"}
    
    return {"prediction": int(pred), "label": labels.get(int(pred), "unknown")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
