FROM python:3.9.18-bullseye
WORKDIR /opt/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-torch.txt .
RUN pip install --no-cache-dir -r requirements-torch-cpu.txt

COPY requirements-cv.txt .
RUN pip install --no-cache-dir -r requirements-cv.txt
