# Complete Setup Guide - MNIST MLOps Pipeline

This guide walks you through setting up the complete MLOps pipeline from scratch, including all components and verification steps.

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Python 3.11** installed
- [ ] **Conda** (Miniconda or Anaconda) installed
- [ ] **Docker** installed and running
- [ ] **Git** installed
- [ ] **Weights & Biases account** (optional, for experiment tracking)

## 🚀 Step-by-Step Setup

### Step 1: Clone and Navigate

```bash
# Clone the repository
git clone <your-repo-url>
cd MLOps

# Verify you're in the right directory
ls -la
# Should show: src/, conf/, environment.yaml, Dockerfile, etc.
```

### Step 2: Environment Setup

```bash
# Create the conda environment
conda env create -f environment.yaml

# Activate the environment
conda activate MNIST_NUM_DETECT

# Verify activation
python --version  # Should show Python 3.11.x
conda list | grep pytorch  # Should show PyTorch packages
```

**Expected Output:**
```
Python 3.11.x
pytorch                   2.x.x
torchvision              0.x.x
torchaudio               2.x.x
```

### Step 3: Install Package in Development Mode

```bash
# Install the package in development mode
pip install -e .

# Verify installation
python -c "import src; print('Package installed successfully')"
```

### Step 4: Weights & Biases Setup (Optional)

```bash
# Copy the environment template
cp env_template.txt .env

# Edit the .env file with your API key
nano .env  # or use your preferred editor
```

**Add your API key to `.env`:**
```
WANDB_API_KEY=your_actual_api_key_here
WANDB_MODE=online
```

**To get your API key:**
1. Go to https://wandb.ai
2. Sign up or log in
3. Go to Settings → API Keys
4. Copy your API key

### Step 5: Verify Basic Setup

```bash
# Test that the main pipeline can be imported
python -c "from src.main import main; print('Main pipeline imported successfully')"

# Test that configuration can be loaded
python -c "import yaml; yaml.safe_load(open('conf/config.yaml')); print('Config loaded successfully')"
```

## 🧪 Testing Each Component

### Test 1: Data Loading

```bash
# Run data loading step
python src/main.py step=data_load

# Verify output
ls -la data/raw/
# Should show: x_train.npy, y_train.npy, x_test.npy, y_test.npy
```

**Expected Output:**
```
INFO - Running pipeline step: data_load
INFO - 🔁 Running data_load step
INFO - ✅ .npy files saved successfully.
```

### Test 2: Data Validation

```bash
# Run data validation
python src/main.py step=data_validation

# Should complete without errors
```

### Test 3: Data Preprocessing

```bash
# Run data preprocessing
python src/main.py step=data_preprocess

# Verify output
ls -la data/processed/
# Should show: train_images.npy, train_labels.npy, test_images.npy, test_labels.npy
```

### Test 4: Model Training (with WandB)

```bash
# Run model training
python src/main.py step=model

# Check WandB dashboard
# Go to: https://wandb.ai/your-username/mlops_mnist_project
```

**Expected Output:**
```
INFO - Running pipeline step: model
INFO - WandB initialized successfully with run name: baseline_run-20250622-120000
INFO - 🔁 Running model step
INFO - Training on cuda for 5 epochs
INFO - Epoch [1/5] | Train Loss: 0.2146 | Train Acc: 0.9362 | Val Loss: 0.0719 | Val Acc: 0.9790
...
INFO - Model saved to models/model.pth
```

### Test 5: Model Evaluation

```bash
# Run evaluation
python src/main.py step=evaluation

# Verify output
ls -la reports/figures/
# Should show: confusion_matrix.png
```

### Test 6: Feature Extraction

```bash
# Run feature extraction
python src/main.py step=features

# Verify output
ls -la reports/embeddings/
# Should show: embeddings.npz, tsne_plot.png, pca_plot.png
```

## 🐳 Docker Setup and Testing

### Build Docker Image

```bash
# Build the inference container
docker build -t mnist-api .

# Verify image was created
docker images | grep mnist-api
```

**Expected Output:**
```
mnist-api    latest    abc123def456    2 minutes ago    2.5GB
```

### Test Docker Container

```bash
# Run the container in background
docker run -d -p 8000:8000 --name mnist-api-container mnist-api

# Check if container is running
docker ps | grep mnist-api

# Test health endpoint
curl http://localhost:8000/health
```

**Expected Health Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": false,
  "gpu_device": null,
  "mlflow_tracking_uri": "",
  "mlflow_model_name": "mnist_model",
  "mlflow_model_stage": "Production"
}
```

### Test API Documentation

1. Open your browser
2. Go to: http://localhost:8000/docs
3. You should see the FastAPI interactive documentation
4. Test the `/health` endpoint from the UI

### Clean Up Docker

```bash
# Stop and remove container
docker stop mnist-api-container
docker rm mnist-api-container
```

## 📊 MLflow Setup (Optional)

### Start MLflow Server

```bash
# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000

# In another terminal, verify it's running
curl http://localhost:5000/health
```

### View MLflow UI

1. Open your browser
2. Go to: http://localhost:5000
3. You should see the MLflow experiment tracking interface

## 🧪 Running Tests

### Install Test Dependencies

```bash
# Install pytest and coverage
conda install -y pytest pytest-cov

# Or if not available in conda
pip install pytest pytest-cov
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_model.py -v
```

**Expected Output:**
```
============================= test session starts ==============================
platform linux -- Python 3.11.x, pytest-7.x.x, pluggy-1.x.x
collected X items

tests/test_model.py::test_cnn_model_creation PASSED                    [ 20%]
tests/test_model.py::test_model_forward_pass PASSED                   [ 40%]
...
============================== X passed in Xs ===============================
```

### View Coverage Report

```bash
# Open coverage report in browser
open htmlcov/index.html  # On macOS
# or
firefox htmlcov/index.html  # On Linux
```

## 🔍 Verification Checklist

After completing all steps, verify:

### Environment
- [ ] `conda activate MNIST_NUM_DETECT` works
- [ ] `python --version` shows 3.11.x
- [ ] All required packages are installed

### Pipeline
- [ ] `python src/main.py step=data_load` works
- [ ] `python src/main.py step=model` works
- [ ] Model files are created in `models/`
- [ ] Reports are generated in `reports/`

### WandB (if configured)
- [ ] `.env` file exists with API key
- [ ] Training logs appear in WandB dashboard
- [ ] Model artifacts are uploaded

### Docker
- [ ] `docker build -t mnist-api .` succeeds
- [ ] `docker run -p 8000:8000 mnist-api` starts
- [ ] `curl http://localhost:8000/health` returns healthy status
- [ ] http://localhost:8000/docs shows API documentation

### Testing
- [ ] `pytest` runs without errors
- [ ] Coverage is above 50%
- [ ] All tests pass

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Install in development mode
pip install -e .
```

#### 2. WandB Connection Issues
```bash
# Error: Failed to initialize WandB
# Solution: Check .env file and internet connection
cat .env  # Should show WANDB_API_KEY=your_key
```

#### 3. Docker Build Failures
```bash
# Error: Build context
# Solution: Ensure you're in the project root directory
pwd  # Should show /path/to/MLOps
ls Dockerfile  # Should exist
```

#### 4. Port Already in Use
```bash
# Error: Port 8000 already in use
# Solution: Use different port or stop existing service
docker run -p 8001:8000 mnist-api  # Use port 8001
```

#### 5. CUDA Issues in Docker
```bash
# Error: NVIDIA Driver not detected
# Solution: Install NVIDIA Container Toolkit or run without GPU
docker run --gpus all -p 8000:8000 mnist-api  # With GPU
```

## 📞 Getting Help

If you encounter issues:

1. **Check the troubleshooting section** in this guide
2. **Review the main README.md** for additional information
3. **Check the logs** for specific error messages
4. **Verify your environment** matches the prerequisites
5. **Create an issue** in the repository with detailed error information

## ✅ Success Criteria

You've successfully set up the MLOps pipeline when:

- [ ] All pipeline steps run without errors
- [ ] Model training completes and saves
- [ ] WandB logs are visible (if configured)
- [ ] Docker container serves the API
- [ ] All tests pass with >50% coverage
- [ ] You can access the FastAPI documentation

**Congratulations! 🎉 Your MLOps pipeline is ready for production use.** 