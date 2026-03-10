FROM python:3.11.14-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HOME=/tmp
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY main.py agent.py /app/
RUN adduser --disabled-password --gecos '' --uid 10001 app && \
    mkdir -p /app/logs && chown -R app:app /app/logs
ENV SANDBOX_AGENT_PORT=8000
EXPOSE 8000
USER app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info"]
