# Docker + MLflow Integration Setup

This guide explains how to use the integrated Docker and MLflow setup for the MLOps pipeline.

## Architecture Overview

The system consists of two main components:
1. **MLflow Tracking Server**: Manages experiments, models, and metadata
2. **FastAPI Application**: Serves predictions using models from MLflow

## Quick Start

### 1. Build the Docker Image

```bash
# Build the Docker image
docker build -t mnist-api .
```

### 2. Run MLflow Pipeline with Docker Backend

```bash
# Run the MLflow pipeline (will use Docker backend)
mlflow run . --backend docker

# Or run specific steps
mlflow run . --backend docker -P step=data_load
mlflow run . --backend docker -P step=preprocess
mlflow run . --backend docker -P step=train
```

### 3. Start MLflow Server

```bash
# Start MLflow server locally
mlflow server --host 0.0.0.0 --port 5000
```

### 4. Run FastAPI App in Docker

```bash
# Run the FastAPI app in Docker
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_MODEL_NAME=mnist_model \
  -e MLFLOW_MODEL_STAGE=Production \
  mnist-api
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction (replace with actual image file)
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_digit_image.png"
```

## Development Workflow

### Option 1: Local Development with MLflow
```bash
# Start MLflow server locally
mlflow server --host 0.0.0.0 --port 5000

# Run pipeline locally
mlflow run . -P step=all

# Test API locally
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Option 2: Docker Development
```bash
# Build the Docker image
docker build -t mnist-api .

# Run MLflow pipeline with Docker backend
mlflow run . --backend docker

# Start MLflow server locally
mlflow server --host 0.0.0.0 --port 5000

# Run FastAPI app in Docker
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  mnist-api
```

## Model Management

### Register a Model in MLflow
```python
import mlflow
import mlflow.pytorch

# Log the model
mlflow.pytorch.log_model(model, "mnist_model")

# Register the model
mlflow.register_model(
    "runs:/<run_id>/model",
    "mnist_model"
)

# Transition to Production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="mnist_model",
    version=1,
    stage="Production"
)
```

### Load Model in API
The FastAPI app automatically loads models from MLflow using:
- Model name: `mnist_model`
- Stage: `Production`
- Tracking URI: `http://host.docker.internal:5000` (Docker) or `http://localhost:5000` (local)

## Environment Variables

### For Docker Container
- `MLFLOW_TRACKING_URI`: MLflow server URL (use `host.docker.internal:5000` for local MLflow)
- `MLFLOW_MODEL_NAME`: Name of the model to load
- `MLFLOW_MODEL_STAGE`: Model stage (Production, Staging, etc.)

### For Local Development
Set these environment variables or use the defaults in the code.

## Troubleshooting

### Common Issues

1. **Model not found in MLflow**
   - Ensure the model is registered and in the correct stage
   - Check the MLflow UI at http://localhost:5000

2. **Docker build fails**
   - Ensure Docker and NVIDIA Docker are installed
   - Check that the Dockerfile is in the root directory

3. **API can't connect to MLflow**
   - Verify MLflow server is running
   - Use `host.docker.internal:5000` instead of `localhost:5000` when running API in Docker

4. **GPU not available**
   - Ensure NVIDIA Docker is installed
   - Check that the GPU is not being used by other processes

### Logs
```bash
# View Docker container logs
docker logs <container_id>

# View MLflow server logs
# Check the terminal where you started mlflow server
```

## Next Steps

1. Train and register your model using the MLflow pipeline
2. Deploy the integrated system to production
3. Set up monitoring and logging
4. Implement CI/CD pipeline for automated deployments 