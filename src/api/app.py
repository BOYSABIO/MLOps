"""MNIST Digit Recognition API using FastAPI and MLflow."""

import io
import os
import sys
import mlflow  # type: ignore
import mlflow.pyfunc  # type: ignore
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from PIL import Image
from src.inference.inference import load_trained_model, predict_digits


# Add the app directory to Python path for imports
sys.path.append("/app")


app = FastAPI(
    title="MNIST Digit Recognition API",
    description="API for recognizing handwritten digits using MLflow",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "mnist_model")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")


@app.on_event("startup")
async def load_model_startup():
    """Load model on app startup."""
    try:
        model = load_trained_model(
            model_path="models/pytorch_mnist_model.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("✅ Local model loaded successfully.")
    except Exception as local_error:
        print(f"⚠️ Failed to load local model: {local_error}")
        print("🔁 Attempting to load model from MLflow...")

        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model = mlflow.pyfunc.load_model(
                f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
            )
            print(
                f"✅ Model loaded from MLflow: "
                f"{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
            )
        except Exception as mlflow_error:
            raise RuntimeError(
                "Failed to load model from local and MLflow: "
                f"{local_error} -> {mlflow_error}"
            ) from mlflow_error

    app.state.model = model


def preprocess_image(image_bytes):
    """Preprocess the uploaded image for prediction."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert(
            "L").resize((28, 28))
        image_array = np.array(image).astype("float32") / 255.0
        return image_array.flatten()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Image preprocessing failed: {str(e)}"
        ) from e


@app.post("/predict")
async def predict_digit_api(file: UploadFile = File(...)):
    """
    Predict the digit in the uploaded image.

    Args:
        file: UploadFile - The image file

    Returns:
        dict: Prediction result
    """
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        contents = await file.read()
        image_array = preprocess_image(contents)

        if hasattr(model, "predict"):
            prediction = model.predict([image_array])
            predicted_digit = int(prediction[0])
            model_source = "MLflow"
        else:
            tensor = torch.tensor(image_array.reshape(1, 28, 28, 1))
            prediction = predict_digits(model, tensor)
            predicted_digit = prediction[0]
            model_source = "Local"

        return {
            "prediction": predicted_digit,
            "message": f"Predicted digit: {predicted_digit}",
            "model_source": model_source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model = app.state.model
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "mlflow_model_name": MLFLOW_MODEL_NAME,
        "mlflow_model_stage": MLFLOW_MODEL_STAGE
    }
