# Weights & Biases (WandB) Integration Setup

This project now supports both MLflow and Weights & Biases for experiment tracking and model management.

## Setup Instructions

### 1. Install Dependencies
The required packages (`wandb` and `python-dotenv`) have been added to:
- `environment.yaml`
- `Dockerfile`

### 2. Get Your WandB API Key
1. Go to [wandb.ai](https://wandb.ai)
2. Create an account or sign in
3. Go to Settings → API Keys
4. Copy your API key

### 3. Configure Environment Variables
1. Copy `env_template.txt` to `.env`:
   ```bash
   cp env_template.txt .env
   ```
2. Edit `.env` and add your API key:
   ```
   WANDB_API_KEY=your_actual_api_key_here
   WANDB_MODE=online
   ```

### 4. Configure WandB Settings
Edit `conf/config.yaml` to customize your WandB project:
```yaml
wandb:
  project: "mlops_mnist_project"  # Your project name
  entity: null  # Your wandb username (optional)
  tags: ["group2", "mnist", "baseline"]
  name: "baseline_run"
  enabled: true  # Set to false to disable wandb
```

## Usage

### Running with WandB
```bash
python src/main.py step=model
```

### Disabling WandB
Set `enabled: false` in the wandb config section or set `WANDB_MODE=disabled` in your `.env` file.

## Features

### What WandB Logs
- **Training Metrics**: Loss and accuracy for each epoch
- **Validation Metrics**: Validation loss and accuracy
- **Model Artifacts**: Saved model files
- **Configuration**: All model hyperparameters

### What MLflow Still Handles
- **Experiment Tracking**: Run organization
- **Model Registry**: Model versioning and staging
- **Pipeline Orchestration**: Step-by-step execution
- **Artifact Storage**: Data and model storage

## Benefits of Dual Integration

1. **MLflow**: Robust experiment tracking and model lifecycle management
2. **WandB**: Beautiful visualizations, real-time monitoring, and collaboration features

Both systems work together seamlessly - you get the best of both worlds! 