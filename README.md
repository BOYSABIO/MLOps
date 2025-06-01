# MNIST Number Detection

This repository provides a modular, production-quality MLOps pipeline for handwritten digit recognition using the MNIST dataset. The system predicts digits from 0 to 9 using a deep learning model built with TensorFlow/Keras, structured for scalability, reproducibility, and clarity.

⸻

🚦 Project Status

Phase 1: MLOps Foundations and Modularization
- ✅ Jupyter notebook translated into modular, testable Python scripts
- ✅ End-to-end pipeline including:
- Data ingestion
- Validation
- Preprocessing
- Model training
- Evaluation
- Inference
- ✅ Config-driven pipeline via config.yaml
- ✅ Unit tests implemented with pytest
- ✅ Reproducible environments using environment.yml

Next Steps:
- Integrate MLflow or W&B for experiment tracking
- Add GitHub Actions for CI/CD automation
- Migrate to Hydra for dynamic configuration management

⸻

<pre><code>## 📁 Repository Structure

```
.
├── README.md
├── config.yaml / config2.yaml        # Configuration files
├── environment.yml                   # Conda environment definition
├── data/
├── models/                           # Saved model weights (.h5, .pth)
├── reports/
├── docs/                             # Guidelines and brainstorm notes
├── notebooks/
├── src/
│   ├── data_load/                    # Data loading utilities
│   ├── data_preprocess/             # Preprocessing steps (e.g. normalization)
│   ├── data_validation/             # Input checks and schema validation
│   ├── features/                    # Feature engineering if needed
│   ├── model/                       # Model definition and training
│   ├── inference/                   # Batch inference logic
│   ├── evaluation/                  # Evaluation metrics
│   ├── utils/                       # Logging configuration
│   └── main.py                      # Pipeline orchestration
├── tests/                            # Unit tests for all modules
```
</code></pre>



⸻

🧠 Problem Description

The task is to recognize handwritten digits from grayscale images (28x28 pixels) in the MNIST dataset. The pipeline trains a Convolutional Neural Network (CNN) to classify each image into one of 10 digit classes (0 through 9).

Model Summary:
- 3 convolutional layers with batch normalization and dropout
- Fully connected dense layers
- Output: softmax for 10-class classification
- Optimizer: Adam | Loss: Categorical Crossentropy
- Accuracy: >99% on training / >98% on validation

⸻

🧪 Pipeline Modules

1. Data Loading (src/data_load/data_loader.py)
- Loads data stored in .npy format
- Reads split configuration (train/test) from YAML config

2. Preprocessing (src/data_preprocess/data_preprocessing.py)
- Normalizes pixel values (0–255 → 0–1)
- Reshapes inputs and one-hot encodes labels

3. Data Validation (src/data_validation/validation.py)
- Validates input format, shape, and missing values
- Raises warnings or errors depending on severity

4. Model Definition (src/model/model.py)
- Defines a CNN using Keras Functional API
- Modular structure for easy architectural changes

5. Evaluation (src/evaluation/evaluation.py)
- Computes performance metrics (accuracy, loss)
- Can be extended with confusion matrices, etc.

6. Inference (src/inference/inference.py)
- Loads model and preprocessing pipeline
- Performs predictions on new data inputs

7. Testing (tests/)
- Unit tests for each pipeline component
- Ensures reliability and simplifies future changes

⸻

⚙️ Configuration and Reproducibility
- config.yaml: Defines data paths, hyperparameters, output paths
- environment.yml: Sets up all dependencies for reproducible runs
- Version-controlled artifacts (models, data, logs)

⸻

🚀 Getting Started
	1.	Set up the environment

conda env create -f environment.yml
conda activate mnist-pipeline


	2.	Run the full pipeline

python -m src.main --config config.yaml --stage all


	3.	Run the unit tests

pytest



⸻

📚 Notes for Teaching and Reuse

This project is designed for academic use, showcasing how to move from notebook-based prototyping to modular, testable ML systems using MLOps best practices.

⸻

👩‍💻 Authors

Project developed as part of an academic MLOps course
MNIST dataset courtesy of Yann LeCun and the NYU team
Inspired by industry-standard MLOps tools and workflows

⸻

📜 License

For educational and non-commercial use only.
See LICENSE for more details.
