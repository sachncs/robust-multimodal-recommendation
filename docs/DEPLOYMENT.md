# Deployment Guide

This guide covers deploying RMR in various environments.

## Local Development

### Quick Start

```bash
# Clone and install
git clone https://github.com/sachin/rmr.git
cd rmr
pip install -e ".[dev]"

# Run demo
python demo.py
```

### Production Setup

1. **Environment Preparation**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install production dependencies
   pip install .
   ```

2. **Data Preparation**
   ```bash
   # Follow data pipeline in docs/SETUP.md
   python -m rmr.data.download
   python -m rmr.data.features
   python -m rmr.data.graph_builder
   python -m rmr.data.masking
   ```

3. **Model Training**
   ```bash
   # Train completion model
   python -m rmr.scripts.train_completion --data-dir data/processed --epochs 100
   
   # Train recommender
   python -m rmr.scripts.train_recommender --data-dir data/processed --epochs 100
   ```

## Cloud Deployment

### AWS

#### EC2 Instance

1. **Launch Instance**
   - Choose Amazon Linux 2 or Ubuntu
   - Instance type: `m5.xlarge` or larger (GPU: `p3.2xlarge`)
   - Security group: Allow SSH (22) and your application port

2. **Setup**
   ```bash
   # Update system
   sudo yum update -y  # Amazon Linux
   # or
   sudo apt update && sudo apt upgrade -y  # Ubuntu
   
   # Install Python
   sudo yum install python3.11 -y  # Amazon Linux
   # or
   sudo apt install python3.11 python3.11-venv -y  # Ubuntu
   
   # Clone and install
   git clone https://github.com/sachin/rmr.git
   cd rmr
   python3.11 -m venv venv
   source venv/bin/activate
   pip install .
   ```

#### ECS (Docker)

See Docker section below.

### Google Cloud Platform

#### Compute Engine

Similar to AWS EC2 setup.

#### AI Platform

For managed training and prediction.

### Azure

#### Virtual Machines

Similar to AWS EC2 setup.

#### Azure ML

For managed ML workflows.

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Expose port (if serving predictions)
EXPOSE 8000

# Default command
CMD ["python", "demo.py"]
```

### Build and Run

```bash
# Build image
docker build -t rmr .

# Run container
docker run -it rmr

# Run with GPU support (requires nvidia-docker)
docker run --gpus all -it rmr
```

### Docker Compose

```yaml
version: '3.8'

services:
  rmr:
    build: .
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
    environment:
      - DEVICE=cuda
      - DATA_PROCESSED_DIR=/app/data/processed
    ports:
      - "8000:8000"
```

## Kubernetes Deployment

### Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rmr
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rmr
  template:
    metadata:
      labels:
        app: rmr
    spec:
      containers:
      - name: rmr
        image: rmr:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1  # For GPU support
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: checkpoint-volume
          mountPath: /app/checkpoints
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: rmr-data-pvc
      - name: checkpoint-volume
        persistentVolumeClaim:
          claimName: rmr-checkpoint-pvc
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rmr-service
spec:
  selector:
    app: rmr
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

## Environment Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Data paths
export DATA_RAW_DIR=data/raw
export DATA_PROCESSED_DIR=data/processed
export DATA_FEATURES_DIR=data/features

# Training
export DEVICE=cuda
export RANDOM_SEED=42
export BATCH_SIZE=512
export LEARNING_RATE=0.001
export EPOCHS=100

# Logging
export LOG_LEVEL=INFO
export DEBUG=false
```

### Configuration Files

- `.env` - Environment variables
- `pyproject.toml` - Project configuration
- `.editorconfig` - Editor settings

## Monitoring & Logging

### Logging

RMR uses Python's built-in logging. Configure log level:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Metrics

Training metrics are output to console. For production monitoring:

1. **TensorBoard**
   ```bash
   pip install tensorboard
   tensorboard --logdir=logs/
   ```

2. **Custom Logging**
   Modify training scripts to log to files or external services.

### Health Checks

For production deployments, implement health checks:

```python
@app.route('/health')
def health():
    return {'status': 'healthy', 'model_loaded': model is not None}
```

## Scaling

### Horizontal Scaling

- Use Kubernetes or Docker Swarm for multiple replicas
- Implement load balancing for prediction requests
- Consider model sharding for large models

### Vertical Scaling

- Use GPU instances for training
- Increase memory for larger datasets
- Use faster CPUs for data preprocessing

## Security

### Best Practices

1. **Secrets Management**
   - Use environment variables for sensitive configuration
   - Never commit secrets to version control
   - Consider using AWS Secrets Manager or similar services

2. **Network Security**
   - Use VPCs for isolated environments
   - Implement proper firewall rules
   - Use HTTPS for API endpoints

3. **Data Security**
   - Encrypt data at rest and in transit
   - Implement proper access controls
   - Regular backups

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Use mixed precision training

2. **Slow Training**
   - Use GPU acceleration
   - Increase batch size (within memory limits)
   - Use data loading workers

3. **Data Loading Errors**
   - Verify data paths
   - Check file permissions
   - Ensure sufficient disk space

## Performance Optimization

### Training

1. **Mixed Precision Training**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   with autocast():
       output = model(input)
       loss = criterion(output, target)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **Data Loading**
   ```python
   DataLoader(dataset, num_workers=4, pin_memory=True)
   ```

### Inference

1. **Model Optimization**
   ```python
   model.eval()
   model = torch.jit.script(model)  # TorchScript
   ```

2. **Batch Inference**
   - Process multiple items simultaneously
   - Use appropriate batch sizes

## Backup & Recovery

### Data Backup

```bash
# Backup data
tar -czf backup.tar.gz data/

# Backup models
tar -czf models.tar.gz checkpoints/
```

### Recovery

```bash
# Restore data
tar -xzf backup.tar.gz

# Restore models
tar -xzf models.tar.gz
```

## Support

For deployment issues:
- Check [FAQ.md](FAQ.md)
- Search [existing issues](https://github.com/sachin/rmr/issues)
- Open a new issue with deployment details
