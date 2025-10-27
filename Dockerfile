# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including dos2unix for line ending conversion
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Fix line endings for Unix compatibility and make executable
RUN dos2unix run && chmod +x run

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV SCORER_MAX_WORKERS=4

# Expose any ports if needed (not required for CLI tool)
# EXPOSE 8080

# Default command
CMD ["bash"]