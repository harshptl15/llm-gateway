FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/srv/models

# CPU-only torch first: the default PyPI wheel drags in ~2GB of CUDA libraries
# the gateway never uses (MiniLM runs fine on CPU at ~5ms per prompt).
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# Bake the embedding model into the image so containers start without network
# access and without a first-request download stall.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"
ENV HF_HUB_OFFLINE=1

# Code last: editing app/ only rebuilds this layer, not the 700MB of deps above.
COPY app ./app

EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --retries=20 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
