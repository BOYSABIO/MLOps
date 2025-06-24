[![CI](https://github.com/BOYSABIO/MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/BOYSABIO/MLOps/actions/workflows/ci.yml)

# MNIST Digit Recognition - MLOps Pipeline

A complete MLOps pipeline for handwritten digit recognition using PyTorch, featuring experiment tracking with MLflow and Weights & Biases, containerized inference with FastAPI, and automated CI/CD testing.

## 🏗️ Project Overview

This project demonstrates a production-ready MLOps pipeline with:
- **Data Pipeline**: Data loading, validation, and preprocessing
- **Model Training**: PyTorch CNN with validation metrics
- **Experiment Tracking**: MLflow + Weights & Biases integration
- **Model Serving**: FastAPI containerized inference
- **Quality Assurance**: Automated testing and CI/CD
- **Professional Logging**: Structured logging across all modules

## 📁 Project Structure

```
MLOps/
├── .github/workflows/          # CI/CD pipeline
├── conf/                       # Hydra configuration
│   └── config.yaml            # Main configuration file
├── data/                       # Data directories
│   ├── raw/                   # Raw MNIST data
│   └── processed/             # Preprocessed data
├── models/                     # Trained models
├── reports/                    # Evaluation outputs
│   ├── figures/               # Plots and visualizations
│   └── embeddings/            # Model embeddings
├── src/                        # Source code
│   ├── api/                   # FastAPI inference service
│   ├── data_load/             # Data loading module
│   ├── data_validation/       # Data validation module
│   ├── data_preprocess/       # Data preprocessing module
│   ├── model/                 # Model training module
│   ├── evaluation/            # Model evaluation module
│   ├── features/              # Feature extraction module
│   ├── inference/             # Inference utilities
│   ├── utils/                 # Shared utilities
│   └── main.py                # Pipeline orchestration
├── tests/                      # Test suite
├── environment.yaml            # Conda environment
├── Dockerfile                  # Container for inference
├── setup.py                    # Package setup
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11**
- **Conda** (for environment management)
- **Docker** (for containerized inference)
- **Git** (for version control)

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd MLOps

# Create and activate conda environment
conda env create -f environment.yaml
conda activate MNIST_NUM_DETECT
```

### 2. Weights & Biases Setup (Optional)

For experiment tracking and visualizations:

```bash
# Copy environment template
cp env_template.txt .env

# Edit .env and add your WANDB_API_KEY
# Get your API key from https://wandb.ai/settings
```

### 3. Run the Pipeline

```bash
# Run the complete pipeline
python src/main.py step=all

# Or run individual steps
python src/main.py step=data_load
python src/main.py step=model
python src/main.py step=evaluation
```

## 📊 Pipeline Steps

### 1. Data Loading (`data_load`)
- Downloads MNIST dataset
- Saves raw data as `.npy` files
- **Output**: `data/raw/` directory

### 2. Data Validation (`data_validation`)
- Validates data integrity and schema
- Checks for missing or corrupted files
- **Output**: Validation reports

### 3. Data Preprocessing (`data_preprocess`)
- Normalizes pixel values (0-1)
- Reshapes data for PyTorch
- One-hot encodes labels
- **Output**: `data/processed/` directory

### 4. Model Training (`model`)
- Trains CNN model with validation split
- Logs metrics to Weights & Biases
- Saves model artifacts
- **Output**: `models/model.pth`

### 5. Model Evaluation (`evaluation`)
- Evaluates model on test set
- Generates confusion matrix
- **Output**: `reports/figures/confusion_matrix.png`

### 6. Feature Extraction (`features`)
- Extracts model embeddings
- Creates t-SNE and PCA visualizations
- **Output**: `reports/embeddings/` directory

### 7. Inference (`inference`)
- Makes predictions on new data
- **Output**: `predictions/prediction.txt`

## 🔧 Configuration

### Main Configuration (`conf/config.yaml`)

```yaml
step: all                    # Pipeline step to run
steps:                       # Available steps
  - data_load
  - data_validation
  - data_preprocess
  - model
  - evaluation
  - features
  - inference

paths:                       # File paths
  raw_data: data/raw
  processed_data: data/processed
  model: models/model.pth
  reports: reports/embeddings
  figures: reports/figures
  predictions: predictions/prediction.txt

model:                       # Model hyperparameters
  batch_size: 32
  epochs: 5
  learning_rate: 0.001
  input_shape: [1, 28, 28]
  num_classes: 10
  val_split: 0.2

wandb:                       # Weights & Biases settings
  project: "mlops_mnist_project"
  entity: null
  tags: ["group2", "mnist", "baseline"]
  name_prefix: "baseline_run"
  enabled: true
```

## 🐳 Docker Inference

### Build the Container

```bash
# Build the inference container
docker build -t mnist-api .
```

### Run the API

```bash
# Run the FastAPI service
docker run -p 8000:8000 mnist-api
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Interactive documentation
# Open http://localhost:8000/docs in your browser
```

### API Endpoints

- **GET `/health`**: Health check and system status
- **POST `/predict`**: Predict digit from uploaded image
- **GET `/docs`**: Interactive API documentation

## 📈 Experiment Tracking

### Weights & Biases

- **Project**: `mlops_mnist_project`
- **Metrics**: Training/validation loss and accuracy
- **Artifacts**: Model files and visualizations
- **Dashboard**: https://wandb.ai/your-username/mlops_mnist_project

### MLflow

- **Tracking Server**: http://localhost:5000
- **Experiments**: Organized by pipeline steps
- **Model Registry**: Versioned model management

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-fail-under=50

# Run specific test file
pytest tests/test_model.py
```

### CI/CD Pipeline

The project includes automated testing via GitHub Actions:
- **Triggers**: Push to main, pull requests
- **Environment**: Conda with `MNIST_NUM_DETECT`
- **Coverage**: Minimum 50% required
- **Status**: Check `.github/workflows/ci.yml`

## 📝 Logging

The project uses structured logging throughout:

```python
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Processing data...")
logger.error("Error occurred", exc_info=True)
```

**Log Format**: `2025-06-22 11:30:15 - src.model.model - INFO - Training on cuda for 5 epochs`

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors in Docker**
   - Ensure all imports use absolute paths
   - Check `sys.path.append('/app')` in containerized modules

2. **Weights & Biases Not Working**
   - Verify `.env` file exists with `WANDB_API_KEY`
   - Check internet connection
   - Set `enabled: false` in config to continue without wandb

3. **MLflow Server Not Found**
   - Start MLflow server: `mlflow server --host 0.0.0.0 --port 5000`
   - Or run without MLflow tracking (local mode)

4. **CUDA Not Available in Docker**
   - Install NVIDIA Container Toolkit
   - Use `--gpus all` flag: `docker run --gpus all -p 8000:8000 mnist-api`

### Environment Issues

```bash
# Recreate environment
conda env remove -n MNIST_NUM_DETECT
conda env create -f environment.yaml

# Update packages
conda env update -f environment.yaml
```

## 📚 Additional Documentation

- **Executive Summary**: [docs/MLOPS_Exec_Summ.pdf](docs/MLOPS_Exec_Summ.pdf) - Project overview and technical summary
- **Complete Setup Guide**: [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) - Step-by-step setup from scratch
- **Production Setup**: [docs/PRODUCTION_SETUP.md](docs/PRODUCTION_SETUP.md) - Production deployment guide

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes and test**: `pytest`
4. **Commit changes**: `git commit -m "Add feature"`
5. **Push to branch**: `git push origin feature-name`
6. **Create pull request**

## 📄 License

This project is part of the MLOps course at IE University.

## 👥 Team

- **Group 2** - MNIST Digit Recognition Pipeline
- **Technologies**: PyTorch, MLflow, Weights & Biases, FastAPI, Docker

---

**For questions or issues, please check the troubleshooting section or create an issue in the repository.**
