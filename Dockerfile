FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
RUN pip install --no-cache-dir uv

COPY . .

# Install Python dependencies using uv sync
RUN uv sync --frozen --no-dev

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

# Give read and write access to the store_creds volume
RUN mkdir -p /app/store_creds \
    && chown -R app:app /app/store_creds \
    && chmod 755 /app/store_creds

USER app

# Expose port (Cloud Run standard is 8080)
EXPOSE 8080
# Expose additional port if PORT environment variable is set to a different value
ARG PORT
EXPOSE ${PORT:-8080}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${PORT:-8080}/health || exit 1'

# Set environment variables for Python startup args
ENV TOOL_TIER=""
ENV TOOLS=""

# Set version from build args
ARG API_VERSION
ENV APP_VERSION=${API_VERSION}

# Use entrypoint for the base command and CMD for args
# CRITICAL: Use .venv/bin/python to run main.py directly from source, not via uv run
# This ensures we're running the actual source code copied into the image,
# not the installed package version which may not match the source
ENTRYPOINT ["/bin/sh", "-c"]
CMD [".venv/bin/python main.py --transport streamable-http ${TOOL_TIER:+--tool-tier \"$TOOL_TIER\"} ${TOOLS:+--tools $TOOLS}"]
