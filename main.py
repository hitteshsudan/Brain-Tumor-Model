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

# ==========================
# CORS Setup
# ==========================
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

print(f"📂 Current directory: {os.getcwd()}")
print(f"📄 Models found: {[f for f in os.listdir('.') if f.endswith('.h5')]}")

# ==========================
# Load Brain Tumor Model
# ==========================
BRAIN_TUMOR_MODEL_PATH = "Brain_Tumor_Model.h5"
brain_tumor_model = None

try:
    brain_tumor_model = tf.keras.models.load_model(BRAIN_TUMOR_MODEL_PATH)
    print(f"✅ Brain Tumor model loaded! Input shape: {brain_tumor_model.input_shape}")
except Exception as e:
    print(f"❌ Error loading Brain Tumor model: {e}")

# ✅ Alphabetical order matches Keras class_indices
brain_tumor_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ==========================
# Load Chest X-ray Model
# ==========================
CHEST_XRAY_MODEL_PATH = "Xray_Model.h5"
chest_xray_model = None

try:
    chest_xray_model = tf.keras.models.load_model(CHEST_XRAY_MODEL_PATH)
    print(f"✅ Chest X-ray model loaded! Input shape: {chest_xray_model.input_shape}")
except Exception as e:
    print(f"⚠️ Chest X-ray model not found: {e}")

# ✅ Alphabetical order: NORMAL=0, PNEUMONIA=1
chest_xray_labels = ["NORMAL", "PNEUMONIA"]


# ==========================
# Helper Functions
# ==========================
def save_uploaded_file(file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"📁 File saved at: {destination}")


def preprocess_image(image_path: str, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size, color_mode="rgb")
    img = image.img_to_array(img)
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
# Brain Tumor Endpoint
# ==========================
@app.post("/upload/")
async def upload_brain_tumor(data: UploadFile = File(...)):
    try:
        print(f"🧠 BRAIN TUMOR endpoint - Received: {data.filename}")

        if brain_tumor_model is None:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Brain Tumor model not loaded",
                    "prediction": None,
                    "confidence": None,
                    "model_type": "Brain Tumor MRI"
                }
            )

        timestamp = int(time.time())
        filename = f"{timestamp}_{data.filename}"
        file_location = os.path.join(UPLOAD_DIR, filename)
        save_uploaded_file(data, file_location)

        img = preprocess_image(file_location)
        print(f"📊 Image shape: {img.shape}")

        predictions = brain_tumor_model.predict(img)
        print(f"📈 Raw predictions: {predictions}")

        predicted_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions)) * 100

        if confidence < 60.0:
            result = "Uncertain"
            message = "Model could not confidently classify. Please consult a doctor."
        else:
            result = brain_tumor_labels[predicted_index]
            if result == "No Tumor":
                message = "No tumor detected in the MRI scan."
            else:
                message = f"{result} tumor detected. Please consult a neurologist immediately."

        print(f"✅ BRAIN TUMOR Prediction: {result}, Confidence: {confidence:.2f}%")

        return JSONResponse(
            content={
                "prediction": result,
                "confidence": f"{confidence:.2f}",
                "filename": filename,
                "model_type": "Brain Tumor MRI",
                "message": message,
                "image_url": f"/uploads/{filename}"
            }
        )

    except Exception as e:
        print(f"❌ BRAIN TUMOR ERROR: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "prediction": None,
                "confidence": None,
                "model_type": "Brain Tumor MRI"
            }
        )


# ==========================
# Chest X-ray Endpoint
# ==========================
@app.post("/upload/chest-xray")
async def upload_chest_xray(data: UploadFile = File(...)):
    try:
        print(f"🫁 CHEST X-RAY endpoint - Received: {data.filename}")

        if chest_xray_model is None:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Chest X-ray model not loaded",
                    "prediction": None,
                    "confidence": None,
                    "model_type": "Chest X-ray"
                }
            )

        timestamp = int(time.time())
        filename = f"{timestamp}_{data.filename}"
        file_location = os.path.join(UPLOAD_DIR, filename)
        save_uploaded_file(data, file_location)

        img = preprocess_image(file_location)
        print(f"📊 Image shape: {img.shape}")

        predictions = chest_xray_model.predict(img)
        print(f"📈 Raw predictions: {predictions}")

        predicted_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions)) * 100

        if confidence < 60.0:
            result = "Uncertain"
            message = "Model could not confidently classify. Please consult a doctor."
        elif chest_xray_labels[predicted_index] == "PNEUMONIA":
            result = "PNEUMONIA"
            message = "Signs of Pneumonia detected. Please consult a doctor immediately."
        else:
            result = "NORMAL"
            message = "No signs of Pneumonia. Chest X-ray appears Normal."

        print(f"✅ CHEST X-RAY Prediction: {result}, Confidence: {confidence:.2f}%")

        return JSONResponse(
            content={
                "prediction": result,
                "confidence": f"{confidence:.2f}",
                "filename": filename,
                "model_type": "Chest X-ray",
                "message": message,
                "image_url": f"/uploads/{filename}"
            }
        )

    except Exception as e:
        print(f"❌ CHEST X-RAY ERROR: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "prediction": None,
                "confidence": None,
                "model_type": "Chest X-ray"
            }
        )


# ==========================
# Health Check
# ==========================
@app.get("/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "brain_tumor_model": "loaded ✅" if brain_tumor_model else "not loaded ❌",
            "chest_xray_model": "loaded ✅" if chest_xray_model else "not loaded ❌",
            "brain_tumor_labels": brain_tumor_labels,
            "chest_xray_labels": chest_xray_labels
        }
    )


# ==========================
# Run FastAPI app
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
