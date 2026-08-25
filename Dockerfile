# Container for the risk-report web app. Runs the Streamlit UI on $PORT.
FROM python:3.13-slim

# fonts help matplotlib/reportlab render cleanly; build tools for any wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# cache/ and snapshots/ should be a mounted volume in production so market
# data and daily snapshots persist across restarts (see DEPLOY.md)
ENV RISK_CACHE_DIR=/data/cache \
    RISK_OUT_DIR=/data/reports \
    RISK_SNAP_DIR=/data/snapshots \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/mpl
# world-writable so hosts that run the container as a non-root user
# (e.g. Hugging Face Spaces) can still write the cache and snapshots
RUN mkdir -p /data/cache /data/reports /data/snapshots && chmod -R 777 /data

EXPOSE 8501
# honour the platform's $PORT if set (Render/Railway/Fly), else 8501
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
