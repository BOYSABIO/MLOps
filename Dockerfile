# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files
COPY environment.yaml .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r <(grep -v "^- " environment.yaml | grep -v "^name:" | grep -v "^channels:" | sed 's/^  - //' | sed 's/^    - //') \
    fastapi==0.109.2 \
    python-multipart==0.0.9 \
    uvicorn==0.27.1 \
    pillow==10.2.0

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
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"] 