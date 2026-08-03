FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake git ninja-build python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements-linux.txt requirements-cluster.txt ./
RUN python3 -m venv "$VIRTUAL_ENV" \
    && python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python3 -m pip install --no-cache-dir -r requirements-linux.txt -r requirements-cluster.txt

COPY configs ./configs
COPY ns3 ./ns3
COPY python ./python
COPY cluster ./cluster
COPY scripts/setup_ns3_linux.sh ./scripts/setup_ns3_linux.sh
RUN chmod +x ./scripts/setup_ns3_linux.sh \
    && ./scripts/setup_ns3_linux.sh

COPY scripts/run_training_linux.sh ./scripts/run_training_linux.sh
RUN chmod +x ./scripts/run_training_linux.sh \
    && mkdir -p runs weights

ENTRYPOINT ["python3", "python/hrllc_tsn_trainer.py"]
