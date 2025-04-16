FROM python:3.11-slim

WORKDIR /chatllm

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /chatllm/

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

EXPOSE 2001

CMD ["python3", "inference.py"]
