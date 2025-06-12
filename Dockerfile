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

# Copy environment files
COPY environment.yaml .
COPY setup.py .

# Configure conda channels
RUN conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda config --set channel_priority flexible

# Create conda environment with CUDA-enabled PyTorch
RUN conda create -n MNIST_NUM_DETECT python=3.10 -y && \
    conda activate MNIST_NUM_DETECT && \
    conda install -y -c pytorch -c nvidia -c conda-forge \
    pytorch=2.0.1 \
    torchvision=0.15.2 \
    torchaudio=2.0.2 \
    cudatoolkit=11.8 \
    numpy=1.26.4 \
    pandas=2.2.3 \
    matplotlib=3.10.3 \
    seaborn=0.13.2 \
    scikit-learn=1.6.1 \
    scipy=1.11.4 \
    pyyaml=6.0.2 \
    pytest=8.3.5 \
    tensorflow=2.13.0 \
    opencv-python=4.11.0.86

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "MNIST_NUM_DETECT", "/bin/bash", "-c"]

# Install additional FastAPI dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.2 \
    python-multipart==0.0.9 \
    uvicorn==0.27.1

# Copy the rest of the application
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY config.yaml .

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["conda", "run", "-n", "MNIST_NUM_DETECT", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"] 