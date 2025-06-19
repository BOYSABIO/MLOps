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

# Copy environment files first to leverage Docker cache
COPY environment.yaml .
COPY setup.py .

# Create conda environment and install dependencies
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda env create -f environment.yaml && \
    conda clean -afy

# Copy the rest of the application
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY config.yaml .

# Set environment variables
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate MNIST_NUM_DETECT && uvicorn src.api.app:app --host 0.0.0.0 --port 8000"] 