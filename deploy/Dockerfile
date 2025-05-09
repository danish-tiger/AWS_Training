# ---- Stage 1: Build ----
    FROM python:3.8-slim AS builder

    # Set working directory
    WORKDIR /app

    # Only copy requirements first to leverage Docker cache
    COPY requirements.txt .

    # Install dependencies into a temporary directory
    RUN pip install --no-cache-dir --target=/dependencies -r requirements.txt


    # ---- Stage 2: Final Image ----
    FROM python:3.8-slim

    # Set working directory
    WORKDIR /app

    # Set environment variables
    ENV PYTHONPATH=/app
    ENV GIT_PYTHON_REFRESH=quiet
    ENV PYTHONPATH="${PYTHONPATH}:/app/src"

    # Copy only dependencies from the builder
    COPY --from=builder /dependencies /usr/local/lib/python3.8/site-packages

    # Now copy source code separately
    COPY . /app

    # Default command to run your pipeline
    CMD ["python", "-m", "scripts.main"]
