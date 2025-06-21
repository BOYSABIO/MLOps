"""
# Build the Docker image
docker build -t mnist-api .

# Run the container
docker run -p 8000:8000 mnist-api
"""

import os
import mlflow
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
from PIL import Image
import io
import mlflow.pyfunc

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

# Load the model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Load model from MLflow
        model = mlflow.pyfunc.load_model(
            f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
        )
        print(f"Model loaded successfully from MLflow: "
              f"{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}")
    except Exception as e:
        print(f"Failed to load model from MLflow: {str(e)}")
        print("Attempting to load local model as fallback...")
        try:
            # Fallback to local model loading
            from src.inference.inference import load_trained_model
            
            model = load_trained_model(
                model_path="models/pytorch_mnist_model.pth",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print("Local model loaded successfully as fallback")
        except Exception as local_error:
            raise RuntimeError(
                f"Failed to load model from MLflow and local fallback: "
                f"{str(e)} -> {str(local_error)}"
            )

def preprocess_image(image_bytes):
    """Preprocess the uploaded image for prediction."""
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        image_array = image_array.astype('float32') / 255.0
        
        # Reshape for MLflow model (flatten to 1D array)
        image_array = image_array.flatten()
        
        return image_array
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Image preprocessing failed: {str(e)}"
        )

@app.post("/predict")
async def predict_digit_api(file: UploadFile = File(...)):
    """
    Predict the digit in the uploaded image using MLflow model.
    
    Args:
        file: UploadFile - The image file containing a handwritten digit
        
    Returns:
        dict: Prediction result with the predicted digit and confidence
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Preprocess the image
        image_array = preprocess_image(contents)
        
        # Make prediction using MLflow model
        if hasattr(model, 'predict'):
            # MLflow pyfunc model
            prediction = model.predict([image_array])
            predicted_digit = int(prediction[0])
        else:
            # Local model fallback
            from src.inference.inference import predict_digits
            tensor = torch.tensor(image_array.reshape(1, 28, 28, 1))
            predictions = predict_digits(model, tensor)
            predicted_digit = predictions[0]
        
        return {
            "prediction": predicted_digit,
            "message": f"Predicted digit: {predicted_digit}",
            "model_source": "MLflow" if hasattr(model, 'predict') else "Local"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": (torch.cuda.get_device_name(0) 
                      if torch.cuda.is_available() else None),
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "mlflow_model_name": MLFLOW_MODEL_NAME,
        "mlflow_model_stage": MLFLOW_MODEL_STAGE
    } 