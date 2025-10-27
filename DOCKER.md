# Docker Containerization Guide

This document explains how to containerize and run the scoring tool using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)
- .env.docker file containing environment variables

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t scorer-tool .
```

### 2. Set up Environment Variables

Copy the environment template:
```bash
cp .env.docker .env
```

Edit `.env` file and add your actual tokens and API keys.

### 3. Run with Docker Compose (Recommended)

```bash
# Start the container
docker-compose up -d

# Access the container shell
docker-compose exec scorer bash

# Inside the container, you can now run commands:
./run install
./run test
./run urls.txt
```

### 4. Alternative: Run with Docker directly

```bash
# Run interactively
docker run -it --env-file .env -v $(pwd):/app scorer-tool

# Run a specific command
docker run --env-file .env -v $(pwd):/app scorer-tool ./run install
```

## Container Structure

- **Base Image**: Python 3.11 slim
- **Working Directory**: `/app`
- **Python Path**: `/app/src`
- **Logs**: Stored in `/app/logs` (mounted as volume)

## Available Commands Inside Container

```bash
# Install dependencies
./run install

# Run tests with coverage
./run test

# Score URLs from a file
./run <url_file>

# Run specific Python modules
python3 src/scorer/cli_updated.py <url_file>

# Run pytest directly
python3 -m pytest tests/
```

## Environment Variables

The following environment variables are required:

- `HF_TOKEN`: Hugging Face token for model/dataset access
- `GITHUB_TOKEN`: GitHub token for repository access
- `GEN_AI_STUDIO_API_KEY`: Gen AI Studio API key
- `GENAI_BASE_URL`: Gen AI Studio base URL
- `GENAI_PATH`: Gen AI Studio API path
- `GENAI_MODEL`: Gen AI Studio model name
- `SCORER_MAX_WORKERS`: Number of worker threads (default: 4)

## Volume Mounts

- Current directory (`/app`): For development and accessing input files
- Logs directory (`/app/logs`): For persistent log storage

## Development Workflow

1. **Start the development container**:
   ```bash
   docker-compose up -d
   ```

2. **Access the container**:
   ```bash
   docker-compose exec scorer bash
   ```

3. **Install dependencies** (first time):
   ```bash
   ./run install
   ```

4. **Run tests**:
   ```bash
   ./run test
   ```

5. **Score URLs**:
   ```bash
   ./run src/scorer/urls.txt
   ```

6. **Stop the container**:
   ```bash
   docker-compose down
   ```

## Production Usage

For production deployments, you might want to:

1. **Create a production Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY src/ ./src/
   COPY run ./
   RUN chmod +x run
   ENV PYTHONPATH=/app/src
   ENTRYPOINT ["./run"]
   ```

2. **Run specific commands**:
   ```bash
   docker run --env-file .env -v /path/to/urls.txt:/app/input.txt scorer-tool input.txt
   ```

## Troubleshooting

### Common Issues

1. **Permission denied on `./run`**:
   ```bash
   chmod +x run
   ```

2. **Module not found errors**:
   Ensure `PYTHONPATH=/app/src` is set in the container.

3. **Environment variables not loaded**:
   Make sure your `.env` file exists and contains all required variables.

4. **Port conflicts**:
   If you need to expose ports, modify the `docker-compose.yml` file.

### Debugging

```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs scorer

# Access container shell for debugging
docker-compose exec scorer bash

# Rebuild image after changes
docker-compose build --no-cache
```

## Multi-stage Build (Optional)

For smaller production images, you can use a multi-stage build:

```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY run ./
RUN chmod +x run
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src
ENTRYPOINT ["./run"]
```