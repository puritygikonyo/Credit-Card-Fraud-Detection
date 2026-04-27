# ── Base image ────────────────────────────────────────────────────────────────
# python:3.9-slim is debian-based with minimal extras — keeps the image small
FROM python:3.9-slim

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential: needed to compile some Python packages (e.g. scikit-learn wheels)
# curl: used in the healthcheck below to ping Streamlit's health endpoint
# --no-install-recommends: skips optional packages, shaves ~50MB off the image
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# All subsequent commands run from here; this is also where your code lives
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements FIRST (before source code) so Docker can cache this layer.
# If only your code changes (not requirements.txt), Docker reuses this cached
# layer and skips the pip install — much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
# Copied after pip install to preserve the caching benefit above.
# data/ and models/ are also copied here as fallback — the docker-compose
# volume mounts will override these at runtime with your local files.
COPY app/    ./app/
COPY data/   ./data/
COPY models/ ./models/
COPY setup.py .

# ── Streamlit configuration ───────────────────────────────────────────────────
# Prevents Streamlit from showing the email prompt on first run,
# and disables telemetry — important for a clean production container.
RUN mkdir -p /root/.streamlit
RUN echo '\
[general]\n\
email = ""\n\
\n\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

# ── Port ──────────────────────────────────────────────────────────────────────
# EXPOSE is documentation — it tells Docker (and readers) which port the app
# uses. The actual binding happens in docker-compose or docker run -p.
EXPOSE 8501

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Docker will mark the container unhealthy if Streamlit stops responding.
# --interval: check every 30s
# --timeout: fail if no response within 10s
# --retries: mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]