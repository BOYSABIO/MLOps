# Use miniconda as base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files
COPY environment.yaml .
COPY setup.py .

# Create conda environment from environment.yaml
RUN conda env create -f environment.yaml

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