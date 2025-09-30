# Use Python base image
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Default: run training then prediction
CMD ["bash", "-c", "python src/train.py && python src/predict.py"]
