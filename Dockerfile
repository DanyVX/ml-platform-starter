FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN python -m pip install --upgrade "setuptools==83.0.0" && python -m pip install .
COPY scripts /app/scripts
COPY eval /app/eval
COPY DATA.md MODEL_CARD.md /app/
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/artifacts /app/results && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "mlp.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
