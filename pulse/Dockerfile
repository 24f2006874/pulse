FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements and install with pip (no uv)
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the project into a real top-level package directory.
# This matches the local layout where `pulse` is imported from its parent.
COPY . /workspace/pulse
ENV PYTHONPATH=/workspace

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "pulse.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
