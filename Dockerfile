# Portfolio Visualiser — Streamlit app image.
# Build:  docker build -t portfolio-visualiser .
# Run:    docker run --rm -p 8501:8501 --env-file .env portfolio-visualiser
# See DEPLOYMENT.md for secrets, networking and access control.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libs: kaleido 0.x (Plotly -> PNG for the PDF report) bundles Chromium
# but needs these shared libraries; fonts keep chart text rendering sane.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libgbm1 \
        libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libxkbcommon0 \
        libx11-xcb1 libasound2 libpangocairo-1.0-0 libgl1 \
        fonts-liberation fontconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so the layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# The app writes temp PDFs / charts to its working directory and (when Supabase
# is not configured) Data/saved_portfolios.json, so /app must be writable by
# the non-root runtime user.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

# Secrets (GEMINI_API_KEY, METABASE_API_KEY, BUILDER_PASSWORD, ...) are NOT baked
# in: provide them as environment variables or mount .streamlit/secrets.toml.
CMD ["streamlit", "run", "Advisory_Tool.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
