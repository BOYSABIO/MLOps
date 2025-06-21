# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Initialize conda for bash shell
RUN conda init bash

# Configure conda channels
RUN conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda config --set channel_priority flexible

# Create base conda environment with Python 3.11
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -n MNIST_NUM_DETECT python=3.11 -y && \
    conda clean -afy

# Install conda packages with compatible versions
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate MNIST_NUM_DETECT && \
    conda install -y -c conda-forge -c pytorch \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    pytorch \
    torchvision \
    torchaudio \
    pyyaml \
    pip \
    pytest

# Install pip packages with compatible versions
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate MNIST_NUM_DETECT && \
    pip install --no-cache-dir \
    tensorflow==2.15.0 \
    keras==2.15.0 \
    mlflow==2.9.2 \
    click \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    opencv-python==4.11.0.86 \
    fastapi==0.104.1 \
    python-multipart==0.0.9 \
    uvicorn==0.27.1

# Copy the application code and models
COPY src/ src/
COPY models/ models/
COPY config.yaml .

# Set environment variables
ENV PYTHONPATH=/app
# These will be overridden when running the container
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_MODEL_NAME="mnist_model"
ENV MLFLOW_MODEL_STAGE="Production"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate MNIST_NUM_DETECT && uvicorn src.api.app:app --host 0.0.0.0 --port 8000"] 