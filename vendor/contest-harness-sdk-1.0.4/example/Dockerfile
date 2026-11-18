# Harness image. Build from sdk/contest_harness with the slim pi bundle:
#   cp /path/to/pi/slim/dist/pi.mjs example/pi.mjs
#   docker build -f example/Dockerfile -t contest-harness-sdk:1.0.4 .
FROM node:24-bookworm-slim AS node
FROM python:3.12-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/contest-harness
COPY pyproject.toml /opt/contest-harness/pyproject.toml
COPY contest_harness /opt/contest-harness/contest_harness
RUN pip install --no-cache-dir /opt/contest-harness
COPY template/harness.py /app/harness.py
COPY example/contest_submission.py /app/contest_submission.py
COPY example/pi.mjs /opt/pi-slim/pi.mjs
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CONTEST_PI=/opt/pi-slim/pi.mjs
WORKDIR /app
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
