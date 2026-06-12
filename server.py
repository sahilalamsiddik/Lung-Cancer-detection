#server.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
from predict_model import predict_image

# FastAPI app
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.post("/predict")
async def predict(modality: str = Form(...), image: UploadFile = File(...)):
    try:
        # Save uploaded image
        save_path = f"uploads/{image.filename}"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Run your model
        result = predict_image(save_path)

        return {
            "status": "success",
            "modality": modality,
            "data": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/")
def home():
    return {"message": "Lung Cancer Prediction Backend Running ✔"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=5000, reload=True)
