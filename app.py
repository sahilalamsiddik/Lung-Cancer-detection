from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from predict_model import predict_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static folder
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")


# 👉 Serve index.html at http://localhost:8000
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

#
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_path = "static/uploaded.jpg"

    with open(img_path, "wb") as f:
        f.write(await file.read())

    result = predict_image(img_path)
    return result


# Serve heatmap image
@app.get("/heatmap")
def heatmap():
    return FileResponse("static/cancer_heatmap.jpg")
