# MLflow Integration Guide

## Overview
This document outlines the steps needed to integrate the MLflow model registry with our MNIST API and how to run the application in different environments.

## Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (for GPU acceleration)
- Access to MLflow server (provided by your colleague)

## Environment Setup

### 1. Environment Variables
Create a `.env` file in the project root with the following content:
```
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000  # Update with your MLflow server URL
MLFLOW_MODEL_NAME=mnist_model
MLFLOW_MODEL_STAGE=Production
```

## Running the Application

### Option 1: Using Default Settings
This method uses the values from your `.env` file:
```bash
docker-compose up
```
- Uses the MLflow server URL specified in `.env`
- Uses default model name and stage
- Good for development and testing

### Option 2: Overriding Environment Variables
This method allows you to override settings at runtime:
```bash
MLFLOW_TRACKING_URI=http://your-mlflow-server:5000 docker-compose up
```
- Overrides the MLflow server URL
- Useful for:
  - Different environments (dev, staging, prod)
  - Testing with different MLflow instances
  - When you need to switch between model versions quickly

## Integration Steps with Colleague's Work

1. **Update MLflow Configuration**
   - Get the correct MLflow server URL from your colleague
   - Update `MLFLOW_TRACKING_URI` in `.env`
   - Verify the model name and stage match their setup

2. **Verify Model Access**
   - Ensure you have access to the MLflow server
   - Verify you can see the model in the MLflow UI
   - Check model versioning and stages

3. **Testing the Integration**
   - Run the API with the new MLflow configuration
   - Test predictions to ensure model loading works
   - Verify GPU acceleration is working

## Troubleshooting

### Common Issues

1. **MLflow Connection Issues**
   - Check if MLflow server is running
   - Verify network connectivity
   - Check if the URL is correct

2. **Model Loading Problems**
   - Verify model name and stage
   - Check if model is registered in MLflow
   - Ensure model format is compatible

3. **GPU Issues**
   - Verify NVIDIA drivers are installed
   - Check if CUDA is properly configured
   - Ensure Docker has GPU access

## Maintenance

### Updating the Model
When a new model version is available:
1. Update `MLFLOW_MODEL_STAGE` in `.env` if needed
2. Restart the container:
```bash
docker-compose down
docker-compose up
```

### Monitoring
- Check container logs: `docker-compose logs -f`
- Monitor GPU usage: `nvidia-smi`
- Check MLflow UI for model metrics

## Security Notes
- Never commit `.env` file to version control
- Use appropriate access controls for MLflow server
- Consider using secrets management for production

## Additional Resources
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)

## Docker Image Versioning

### Versioning Strategy
We use semantic versioning for our Docker images: `MAJOR.MINOR.PATCH`
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Building Versioned Images
```bash
# Build with version tag
docker build -t mnist-api:1.0.0 -f Dockerfile.mlflow .

# Build with both version and latest tag
docker build -t mnist-api:1.0.0 -t mnist-api:latest -f Dockerfile.mlflow .
```

### Pushing to Container Registry
```bash
# Tag for your registry (replace with your registry URL)
docker tag mnist-api:1.0.0 your-registry/mnist-api:1.0.0
docker tag mnist-api:latest your-registry/mnist-api:latest

# Push to registry
docker push your-registry/mnist-api:1.0.0
docker push your-registry/mnist-api:latest
```

### Using Versioned Images in docker-compose
Update your `docker-compose.yml` to use specific versions:
```yaml
services:
  mnist-api:
    image: your-registry/mnist-api:1.0.0  # Use specific version
    # ... rest of your configuration
```

### Best Practices
1. **Always use specific versions in production**
   - Avoid using `latest` tag in production
   - Makes rollbacks easier
   - Ensures consistency across environments

2. **Version Naming Convention**
   - Use semantic versioning
   - Include build date for development versions: `1.0.0-20240315`
   - Use git commit hash for traceability: `1.0.0-a1b2c3d`

3. **Registry Organization**
   - Use different registries for different environments
   - Example structure:
     ```
     dev-registry/mnist-api:1.0.0
     staging-registry/mnist-api:1.0.0
     prod-registry/mnist-api:1.0.0
     ```

4. **Version Control Integration**
   - Tag Docker images with git tags
   - Include version in commit messages
   - Document version changes in CHANGELOG.md

### Example Workflow
1. Make changes to code
2. Update version in documentation
3. Build and tag new version:
   ```bash
   docker build -t mnist-api:1.0.1 -f Dockerfile.mlflow .
   ```
4. Push to registry:
   ```bash
   docker tag mnist-api:1.0.1 your-registry/mnist-api:1.0.1
   docker push your-registry/mnist-api:1.0.1
   ```
5. Update docker-compose.yml with new version
6. Deploy using docker-compose:
   ```bash
   docker-compose pull  # Pull new version
   docker-compose up -d  # Deploy
   ```

### Rollback Procedure
If you need to rollback to a previous version:
1. Update docker-compose.yml to use previous version
2. Deploy:
   ```bash
   docker-compose pull
   docker-compose up -d
   ``` 