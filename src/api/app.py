"""
# Build the Docker image
docker build -t mnist-api .

# Run the container
docker run -p 8000:8000 mnist-api
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
from PIL import Image
import io

from src.model.model import CNNModel
from src.inference.inference import load_trained_model, predict_digit

app = FastAPI(
    title="MNIST Digit Recognition API",
    description="API for recognizing handwritten digits using a trained CNN model",
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

# Load the model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = load_trained_model(
            model_path="models/pytorch_mnist_model.pth",
            device="cpu"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

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
        
        # Convert to tensor
        tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)
        
        return tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

@app.post("/predict")
async def predict_digit_api(file: UploadFile = File(...)):
    """
    Predict the digit in the uploaded image.
    
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
        tensor = preprocess_image(contents)
        
        # Make prediction
        prediction = predict_digit(model, tensor)
        
        return {
            "prediction": int(prediction),
            "message": f"Predicted digit: {prediction}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None} 