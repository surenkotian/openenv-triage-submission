FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Create non-root user for security (required for some HF Spaces configs)
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

ENV PATH="/home/appuser/.local/bin:$PATH"

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --no-install-project --no-dev

COPY --chown=appuser:appuser . .

# Expose Hugging Face default port
EXPOSE 7860

# Define labels for Hugging Face space
LABEL tags="openenv"

# Run OpenEnv app using uv
CMD ["uv", "run", "server"]
