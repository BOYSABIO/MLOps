# Production Setup Guide

This guide explains how to deploy the MLOps pipeline to production using Docker containers.

## Production Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLflow        │    │   FastAPI       │    │   External      │
│   Server        │◄───┤   App           │◄───┤   Clients       │
│   (Container)   │    │   (Container)   │    │   (Web/Mobile)  │
│   Port 5000     │    │   Port 8000     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start (Production)

### 1. Build and Deploy Everything

```bash
# Build and start all services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 2. Access Services

- **MLflow UI**: http://localhost:5000
- **FastAPI App**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 3. Test the System

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_digit_image.png"
```

## Production Features

### ✅ **Containerized Services**
- Both MLflow and FastAPI run in Docker containers
- Consistent environments across all deployments
- Easy scaling and replication

### ✅ **Persistent Storage**
- MLflow data persists in Docker volumes
- Model artifacts stored in `mlflow_data` volume
- Database stored in `mlflow_db` volume

### ✅ **Network Isolation**
- Services communicate via internal Docker network
- External access only through exposed ports
- Secure inter-service communication

### ✅ **GPU Support**
- NVIDIA Docker support for GPU acceleration
- Automatic GPU detection and utilization
- Fallback to CPU if GPU unavailable

## Deployment Options

### Option 1: Local Production (Docker Compose)
```bash
# Single machine deployment
docker-compose up -d
```

### Option 2: Cloud Deployment (Kubernetes)
```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      containers:
      - name: mlflow
        image: your-registry/mlflow:latest
        ports:
        - containerPort: 5000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mnist-api
spec:
  replicas: 3  # Scale to multiple instances
  selector:
    matchLabels:
      app: mnist-api
  template:
    metadata:
      labels:
        app: mnist-api
    spec:
      containers:
      - name: api
        image: your-registry/mnist-api:latest
        ports:
        - containerPort: 8000
```

### Option 3: Cloud Services (AWS/GCP/Azure)
```bash
# Deploy to cloud container services
# AWS ECS, Google Cloud Run, Azure Container Instances
```

## Environment Variables

### MLflow Server
```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns
```

### FastAPI App
```bash
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_MODEL_NAME=mnist_model
MLFLOW_MODEL_STAGE=Production
```

## Monitoring and Logging

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mlflow-server
docker-compose logs -f mnist-api
```

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# MLflow health
curl http://localhost:5000/health
```

## Scaling

### Scale API Instances
```bash
# Scale to 3 API instances
docker-compose up -d --scale mnist-api=3
```

### Load Balancer Setup
```yaml
# nginx.conf for load balancing
upstream api_servers {
    server mnist-api:8000;
    server mnist-api:8001;
    server mnist-api:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_servers;
    }
}
```

## Security Considerations

### 1. **Network Security**
- Use internal Docker networks
- Expose only necessary ports
- Implement API authentication

### 2. **Data Security**
- Encrypt sensitive data
- Use secure storage volumes
- Implement access controls

### 3. **API Security**
```python
# Add authentication to FastAPI
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict_digit_api(
    file: UploadFile = File(...),
    token: str = Depends(security)
):
    # Verify token
    if not verify_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    # ... rest of prediction logic
```

## Backup and Recovery

### Backup MLflow Data
```bash
# Backup volumes
docker run --rm -v mlflow_data:/data -v $(pwd):/backup alpine tar czf /backup/mlflow_data_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v mlflow_data:/data -v $(pwd):/backup alpine tar xzf /backup/mlflow_data_backup.tar.gz -C /data
```

## CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build and push Docker images
      run: |
        docker build -t your-registry/mlflow:latest .
        docker build -t your-registry/mnist-api:latest .
        docker push your-registry/mlflow:latest
        docker push your-registry/mnist-api:latest
    
    - name: Deploy to production
      run: |
        ssh user@production-server "cd /app && docker-compose pull && docker-compose up -d"
```

## Troubleshooting

### Common Issues

1. **Services can't communicate**
   ```bash
   # Check network
   docker network ls
   docker network inspect mlops_mlops-network
   ```

2. **Volumes not persisting**
   ```bash
   # Check volumes
   docker volume ls
   docker volume inspect mlops_mlflow_data
   ```

3. **GPU not available**
   ```bash
   # Check NVIDIA Docker
   nvidia-docker run --rm nvidia/cuda:11.8-base nvidia-smi
   ```

### Performance Optimization

1. **Use GPU acceleration**
2. **Scale API instances**
3. **Implement caching**
4. **Use CDN for static assets**

## Next Steps

1. **Set up monitoring** (Prometheus, Grafana)
2. **Implement logging** (ELK Stack)
3. **Add authentication** (JWT, OAuth)
4. **Set up CI/CD** (GitHub Actions, Jenkins)
5. **Configure alerts** (Slack, Email)
6. **Performance testing** (Load testing) 