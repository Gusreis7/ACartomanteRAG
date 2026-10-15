FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU para embeddings/reranker; geração via Ollama local ou Bedrock.
COPY requirements-runtime.lock ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-runtime.lock \
    && pip check \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /home/app/.cache/huggingface /app/reports \
    && chown -R app:app /home/app/.cache /app/reports

COPY --chown=app:app src ./src
COPY --chown=app:app config/config.yaml config/config.aws.yaml ./config/

USER 10001:10001
EXPOSE 8000

# Um worker mantém o limite de geração por processo e evita duplicar modelos.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers", "--no-server-header", "--no-access-log", "--timeout-keep-alive", "5"]

# Imagem opcional para os jobs existentes. Não faz parte do serviço HTTP.
FROM runtime AS tools
USER root
COPY requirements.txt requirements-runtime.txt ./
RUN pip install --no-cache-dir -c requirements-runtime.lock -r requirements.txt \
    && pip check \
    && chown app:app /app
COPY --chown=app:app benchmark ./benchmark
COPY --chown=app:app AED/*.csv ./AED/
USER 10001:10001
CMD ["python", "-m", "src.generation.cli", "Quem é Capitu?"]
