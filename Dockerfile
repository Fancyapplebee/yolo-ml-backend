FROM python:3.10-slim

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV WEIGHTS_PATH=weights/best.pt

# Cloud Run expects port 8080
EXPOSE 8080
CMD ["label-studio-ml", "start", ".", "--port", "8080", "--host", "0.0.0.0"]