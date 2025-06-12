# MNIST Digit Classifier - MLOps Pipeline (PyTorch)

This project implements a full MLOps pipeline for classifying handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN) built with PyTorch.

---

## Project Structure

```
├── data/
│   ├── raw/                  # Raw input data
│   ├── processed/            # Preprocessed train/test splits
│   └── inference/            # New data for inference and prediction results
├── models/                   # Saved PyTorch models
├── reports/
│   ├── figures/              # Evaluation plots (confusion matrix, t-SNE, PCA)
│   └── embeddings/           # Saved model embeddings
├── src/
│   ├── data_load/            # Load and prepare MNIST data
│   ├── data_preprocess/      # Preprocessing functions (normalize, reshape, encode)
│   ├── data_validation/      # Data schema and integrity checks
│   ├── model/                # PyTorch model and training logic
│   ├── evaluation/           # Accuracy and confusion matrix evaluation
│   ├── features/             # Feature extraction and visualizations
│   ├── inference/            # Inference scripts for new input
│   └── utils/                # Logging configuration
├── src/draw_and_infer.py     # Create new data, save & predict
├── src/main.py               # Entry-point to run the pipeline
├── config.yaml               # Model and pipeline configuration
└── README.md                 # This file
```

---

## Getting Started

### Step 1: Create and activate the environment

```
conda env create -f environment.yaml
conda activate MNIST_NUM_DETECT
```

### Step 2: Run the pipeline

```bash
# Full run (data + train + evaluate)
python -m src.main --stage all --config config.yaml

# Only data processing
python -m src.main --stage data --config config.yaml

# Only model training (if preprocessed data exists)
python -m src.main --stage train --config config.yaml

# Inference with live drawing window
python -m src.main --stage infer --config config.yaml
```

---

## Draw and Predict Digits

When you run `--stage infer`, a window will open where you can draw a digit.

- Press **Space** to predict.
- Press **C** to clear.
- Press **ESC** to exit.

Predictions and drawing are logged and optionally saved.

---

## Visualizations

- **Confusion Matrix:** `reports/figures/confusion_matrix.png`
- **t-SNE Plot:** `reports/figures/tsne_plot.png`
- **PCA Plot:** `reports/figures/pca_plot.png`

---

## Model

- CNN implemented in PyTorch with:
  - 3 Conv2D layers + BatchNorm
  - Dropout for regularization
  - Flatten + Fully connected layers
  - Optimized for GPU use (if available)

---

## Outputs

- `models/pytorch_mnist_model.pth` – saved PyTorch model
- `reports/embeddings/embeddings.npz` – feature embeddings for visual analysis
- `data/inference/` – input/output for new predictions

---
