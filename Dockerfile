FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Build the medical codes database
RUN python scripts/build_code_db.py

# Create directories for SQLite fallback
RUN mkdir -p data/codes data/rules data/guidelines data/sample_claims

# Expose port
EXPOSE 8000

# Run with start.py which handles env loading
CMD ["python", "start.py"]
