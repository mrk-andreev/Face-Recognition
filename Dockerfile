FROM python:3.9.18-bullseye
WORKDIR /opt/app

COPY requirements.txt .
COPY requirements-torch.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-torch.txt
