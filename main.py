import os
import shutil
import time
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image

# ==========================
# FastAPI Setup
# ==========================
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ✅ ADD CORS for React
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ==========================
# DEBUG: Show current directory and files
# ==========================
print(f"📂 Current directory: {os.getcwd()}")
print(f"📄 Files in directory: {[f for f in os.listdir('.') if f.endswith('.h5')]}")

# ==========================
# Load Brain Tumor Model
# ==========================
BRAIN_TUMOR_MODEL_PATH = "Brain_Tumor_Model.h5"
brain_tumor_model = None

print(f"🔍 Looking for Brain Tumor model at: {os.path.abspath(BRAIN_TUMOR_MODEL_PATH)}")
print(f"📋 File exists: {os.path.exists(BRAIN_TUMOR_MODEL_PATH)}")

try:
    brain_tumor_model = tf.keras.models.load_model(BRAIN_TUMOR_MODEL_PATH)
    print(f"✅ Loaded Brain Tumor model successfully! Input shape: {brain_tumor_model.input_shape}")
except Exception as e:
    print(f"❌ Error loading Brain Tumor model: {e}")

brain_tumor_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ==========================
# Load Chest X-ray Model
# ==========================
CHEST_XRAY_MODEL_PATH = "Xray_Model.h5"
chest_xray_model = None

print(f"🔍 Looking for Chest X-ray model at: {os.path.abspath(CHEST_XRAY_MODEL_PATH)}")
print(f"📋 File exists: {os.path.exists(CHEST_XRAY_MODEL_PATH)}")

try:
    chest_xray_model = tf.keras.models.load_model(CHEST_XRAY_MODEL_PATH)
    print(f"✅ Loaded Chest X-ray model successfully! Input shape: {chest_xray_model.input_shape}")
except Exception as e:
    print(f"⚠️ Chest X-ray model not found: {e}")

chest_xray_labels = ["Normal", "Pneumonia"]

# ==========================
# Helper functions
# ==========================
def save_uploaded_file(file: UploadFile, destination: str):
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"📁 File saved at: {destination}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        raise e

def preprocess_image_for_model(image_path: str, model, target_size=(224, 224)):
    if model is None:
        raise ValueError("Model not loaded")

    input_channels = model.input_shape[-1]
    color_mode = "grayscale" if input_channels == 1 else "rgb"

    img = image.load_img(image_path, target_size=target_size, color_mode=color_mode)
    img = image.img_to_array(img)

    if img.shape[-1] == 1 and input_channels == 3:
        img = np.repeat(img, 3, axis=-1)

    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# ==========================
# Routes
# ==========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ==========================
# Brain Tumor MRI Endpoint
# ==========================
@app.post("/upload/")
async def upload_brain_tumor(data: UploadFile = File(...)):
    try:
        print(f"🧠 BRAIN TUMOR endpoint - Received file: {data.filename}")

        if brain_tumor_model is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Brain Tumor model not loaded. Check if Brain_Tumor_Model.h5 exists."}
            )

        # Save uploaded file
        timestamp = int(time.time())
        filename = f"{timestamp}_{data.filename}"
        file_location = os.path.join(UPLOAD_DIR, filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(data.file, buffer)
        print(f"📁 File saved at: {file_location}")

        # Preprocess image
        img = image.load_img(file_location, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        print(f"📊 Image shape: {img.shape}")

        # Predict
        predictions = brain_tumor_model.predict(img)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions)) * 100
        result = brain_tumor_labels[predicted_index]
        
        if confidence < 50.0:
            result = "Uncertain"

        print(f"✅ BRAIN TUMOR Prediction: {result}, Confidence: {confidence:.2f}%")

        # ✅ Return JSON for React
        return JSONResponse(
            content={
                "prediction": result,
                "confidence": f"{confidence:.2f}",
                "filename": filename,
                "model_type": "Brain Tumor MRI"
            }
        )

    except Exception as e:
        print(f"❌ BRAIN TUMOR UPLOAD ERROR: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# ==========================
# Chest X-ray Endpoint
# ==========================
@app.post("/upload/chest-xray")
async def upload_chest_xray(data: UploadFile = File(...)):
    try:
        print(f"🫁 CHEST X-RAY endpoint - Received file: {data.filename}")

        if chest_xray_model is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Chest X-ray model not loaded. Check if Xray_Model.h5 exists."}
            )

        # Save uploaded file
        timestamp = int(time.time())
        filename = f"{timestamp}_{data.filename}"
        file_location = os.path.join(UPLOAD_DIR, filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(data.file, buffer)
        print(f"📁 File saved at: {file_location}")

        # Preprocess image
        img = image.load_img(file_location, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        print(f"📊 Image shape: {img.shape}")

        # Predict
        predictions = chest_xray_model.predict(img)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions)) * 100
        result = chest_xray_labels[predicted_index]
        
        if confidence < 50.0:
            result = "Uncertain"

        print(f"✅ CHEST X-RAY Prediction: {result}, Confidence: {confidence:.2f}%")

        return JSONResponse(
            content={
                "prediction": result,
                "confidence": f"{confidence:.2f}",
                "filename": filename,
                "model_type": "Chest X-ray"
            }
        )

    except Exception as e:
        print(f"❌ CHEST X-RAY UPLOAD ERROR: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# ==========================
# Health Check Endpoint
# ==========================
@app.get("/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "brain_tumor_model": "loaded ✅" if brain_tumor_model else "not loaded ❌",
            "chest_xray_model": "loaded ✅" if chest_xray_model else "not loaded ❌"
        }
    )

# ==========================
# Run FastAPI app
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Port 8001
