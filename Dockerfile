# ComfyUI Agent — Production Dockerfile
# Multi-stage build: build plugin UI → run agent server

# Stage 1: Build plugin UI
FROM node:20-slim AS ui-builder
WORKDIR /build
COPY comfyui-plugin/ui/package.json comfyui-plugin/ui/package-lock.json ./
RUN npm ci --ignore-scripts
COPY comfyui-plugin/ui/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.12-slim AS runtime
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

# Copy plugin (with built UI)
COPY comfyui-plugin/ comfyui-plugin/
COPY --from=ui-builder /build/dist/ comfyui-plugin/ui/dist/

# Copy default config
COPY config.yaml config.yaml

# Data directory for SQLite + logs
RUN mkdir -p data

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:5200/api/health || exit 1

EXPOSE 5200

# Run agent server
CMD ["python", "-m", "comfyui_agent"]
