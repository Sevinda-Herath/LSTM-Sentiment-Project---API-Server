# Stock API Docker Setup

This guide explains how to run the Stock API application using Docker.

## Prerequisites

- Docker
- Docker Compose

## Quick Start

### Using Docker Compose (Recommended)

1. Navigate to the docker directory:
```bash
cd docker
```

2. Build and start the container:
```bash
docker-compose up --build
```

3. The API will be available at `http://localhost:8000`

4. To run in detached mode:
```bash
docker-compose up -d --build
```

5. To stop the container:
```bash
docker-compose down
```

### Using Docker directly

1. Build the image (from the project root):
```bash
docker build -f docker/Dockerfile -t stock-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/sentiments:/app/sentiments \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/model-metrics-charts:/app/model-metrics-charts \
  stock-api
```

### Quick Test

Run the test script from the docker directory:
```bash
cd docker
./test-docker.sh
```

### Using the Convenience Script (from project root)

You can also use the convenience script from the project root:
```bash
# Build and start the container
./docker.sh build && ./docker.sh start

# View logs
./docker.sh logs

# Stop the container
./docker.sh stop

# Access container shell
./docker.sh shell

# Run tests
./docker.sh test
```

## API Endpoints

### Health Check
- `GET /` - Health check endpoint

### Sentiment Analysis
- `GET /sentiment_summary/{symbol}` - Get sentiment summary for a stock symbol
- `GET /sentiment_chart/{symbol}` - Get sentiment chart image for a stock symbol

### LSTM Model Metrics
- `GET /metrics/lstm/{symbol}` - Get LSTM model metrics for a stock symbol
- `GET /metrics/lstm/chart/tsp/{symbol}` - Get LSTM test set predictions chart
- `GET /metrics/lstm/chart/tl/{symbol}` - Get LSTM training loss chart

### LSTM Sentiment Model Metrics
- `GET /metrics/lstm_sentiment/{symbol}` - Get LSTM sentiment model metrics for a stock symbol
- `GET /metrics/lstm_sentiment/chart/tsp/{symbol}` - Get LSTM sentiment test set predictions chart
- `GET /metrics/lstm_sentiment/chart/tl/{symbol}` - Get LSTM sentiment training loss chart

### Predictions
- `GET /predict/lstm?symbol={symbol}&days={days}` - Get LSTM price prediction (days parameter optional, default: 60)
- `GET /predict/lstm_sentiment?symbol={symbol}&days={days}` - Get LSTM sentiment-based price prediction (days parameter optional, default: 60)

### Chatbot (Ollama + Gemma 2B)
- `POST /chat` - Chat with a local LLM grounded on your CSVs and optional live prices
  - Requires a running Ollama server on the host (default http://localhost:11434) with `gemma2:2b` pulled.
  - If Ollama is on a different host/port, set `OLLAMA_BASE_URL` env var for the API container.

Tip: When running Docker on Linux, the container can access the host via `host.docker.internal` (may require Docker 20.10+). To point the API to the host Ollama, set:

```yaml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Example Usage
```bash
# Health check
curl http://localhost:8000/

# Get sentiment summary for Apple stock
curl http://localhost:8000/sentiment_summary/AAPL

# Get LSTM prediction for Apple stock
curl http://localhost:8000/predict/lstm?symbol=AAPL&days=30

# Get LSTM sentiment prediction for Apple stock
curl http://localhost:8000/predict/lstm_sentiment?symbol=AAPL&days=30

# Get LSTM model metrics for Microsoft
curl http://localhost:8000/metrics/lstm/MSFT

# Get sentiment chart for Google
curl http://localhost:8000/sentiment_chart/GOOGL

# Chatbot example (outside container)
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Summarize latest predictions for AAPL and MSFT.",
    "symbols": ["AAPL", "MSFT"]
  }'
```

### Supported Stock Symbols
The following stock symbols are currently supported:
- `AAPL` - Apple Inc.
- `MSFT` - Microsoft Corporation
- `GOOGL` - Alphabet Inc. (Google)
- `AMZN` - Amazon.com Inc.
- `META` - Meta Platforms Inc.
- `BABA` - Alibaba Group Holding Ltd.
- `JD` - JD.com Inc.
- `T` - AT&T Inc.
- `005930.KS` - Samsung Electronics (Korea)
- `2317.TW` - Hon Hai Precision Industry (Taiwan)

## Data Persistence

The following directories are mounted as volumes to persist data:
- `datasets/` - Stock data
- `models/` - Trained ML models
- `results/` - Prediction results
- `sentiments/` - Sentiment analysis results
- `logs/` - Application logs
- `model-metrics-charts/` - Model performance charts

## Scheduler

The application includes a background scheduler that runs daily at 3:30 AM UTC (9:00 AM Sri Lanka time) to:
1. Download new stock data
2. Generate sentiment analysis
3. Save predictions

## Troubleshooting

### Check container logs
```bash
# From docker directory
docker-compose logs -f stock-api

# Or using convenience script from project root
./docker.sh logs
```

### Access container shell
```bash
# From docker directory
docker-compose exec stock-api /bin/bash

# Or using convenience script from project root
./docker.sh shell
```

### Rebuild container
```bash
# From docker directory
docker-compose down
docker-compose up --build

# Or using convenience script from project root
./docker.sh stop
./docker.sh build && ./docker.sh start
```

### Container Status
```bash
# Check if container is running
docker ps

# Check container resource usage
docker stats stock-api-container
```
