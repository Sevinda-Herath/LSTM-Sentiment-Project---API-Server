# Stock Prediction API

A comprehensive stock prediction API that uses LSTM neural networks with sentiment analysis to predict stock prices. The API provides real-time predictions, sentiment analysis, and model performance metrics for various stock symbols.

## 🚀 Features

- **Stock Price Prediction**: LSTM-based prediction models with and without sentiment analysis
- **Sentiment Analysis**: Real-time sentiment analysis of stock-related news
- **Model Metrics**: Comprehensive performance metrics and visualizations
- **Automated Data Updates**: Daily scheduled data collection and model updates
- **RESTful API**: Easy-to-use REST endpoints for all functionalities
- **Docker Support**: Containerized deployment for easy setup and scalability

## 📈 Supported Stock Symbols

The API currently supports the following stock symbols:
- **US Stocks**: `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`, `T`
- **Chinese Stocks**: `BABA`, `JD`
- **Korean Stocks**: `005930.KS` (Samsung Electronics)
- **Taiwanese Stocks**: `2317.TW` (Hon Hai Precision Industry)

## 🛠️ Quick Start

### Option 1: Docker (Recommended)

```bash
# Install  Docker
sudo apt  install docker-compose apt-utils

#Add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker

# Quick start with Docker
./docker.sh build && ./docker.sh start

# Or manually with docker-compose
cd docker
docker-compose up --build -d

# View logs
./docker.sh logs

# Stop container
./docker.sh stop
```

### Option 2: Manual Setup

```bash
# Install dependencies
sudo apt update && sudo apt install python3-pip -y
sudo apt install python3.12-venv

# Create virtual environment
python3 -m venv .env
source .env/bin/activate

# Install Python packages
pip3 install -r requirements.txt

# Run the application
bash run.sh

# (Optional) Install Ollama and Gemma 2B locally for the /chat endpoint
bash scripts/install_ollama.sh
```

### Option 3: Auto-run with tmux

```bash
# Install tmux
sudo apt install tmux

# Start in tmux session
tmux
bash run.sh

# Detach from session (Ctrl+B then D)
# Kill session when done
tmux kill-session -t 0
```

## 🏃 Running the API with systemd

Create a `run.sh` script to launch your API:

```bash
#!/bin/bash
cd /home/stock-api
source .env/bin/activate
export PYTHONPATH=$(pwd)
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Make it executable:

```bash
chmod +x /home/stock-api/run.sh
```

---

## ⚙️ Create the `stock-api.service` File

Open the systemd service file for editing:

```bash
sudo nano /etc/systemd/system/stock-api.service
```

Paste the following content:

```ini
[Unit]
Description=Stock Market Prediction API (FastAPI with Uvicorn + venv)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/stock-api
ExecStart=/bin/bash /home/stock-api/run.sh
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

---

## 🚦 Reload & Start the Service

Reload systemd and start your service:

```bash
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable stock-api.service
sudo systemctl start stock-api.service
```

To view logs:

```bash
journalctl -u stock-api.service -f
```

---

## 🌐 Serve a Static Website with Nginx

### 1️⃣ Install Nginx

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install nginx -y
```

**CentOS/RHEL:**
```bash
sudo yum install nginx -y
sudo systemctl enable nginx
```

---

### 2️⃣ Place Your Website Files

**Default directory:**
```bash
sudo rm -rf /var/www/html/*
sudo cp -r /path/to/your/website/* /var/www/html/
```

**Custom directory:**
```bash
sudo mkdir -p /var/www/mywebsite
sudo cp -r /path/to/your/website/* /var/www/mywebsite/
```

---

### 3️⃣ Configure Nginx

**Edit the default config:**
```bash
sudo nano /etc/nginx/sites-available/default
```
Replace the `server` block with:
```nginx
server {
    listen 80 default_server;
    listen [::]:80 default_server;

    root /var/www/html;  # or /var/www/mywebsite
    index index.html;

    server_name _;

    location / {
        try_files $uri $uri/ =404;
    }
}
```

**Or create a new config:**
```bash
sudo nano /etc/nginx/sites-available/mywebsite
```
```nginx
server {
    listen 80;
    server_name yourdomain.com;  # or your server's IP

    root /var/www/mywebsite;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}
```
Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/mywebsite /etc/nginx/sites-enabled/
```

---

### 4️⃣ Test & Restart Nginx

```bash
sudo nginx -t
sudo systemctl restart nginx
```

---

### 5️⃣ Adjust Firewall (if needed)

```bash
sudo ufw allow 'Nginx Full'
```

---

### 6️⃣ View Your Website

Open your browser and visit:  
`http://<your-server-ip>` or `http://yourdomain.com`

---

## 📡 API Endpoints

### Health Check
- `GET /` - API health check

### Sentiment Analysis
- `GET /sentiment_summary/{symbol}` - Get sentiment summary for a stock
- `GET /sentiment_chart/{symbol}` - Get sentiment chart image

### LSTM Model Metrics
- `GET /metrics/lstm/{symbol}` - Get LSTM model performance metrics
- `GET /metrics/lstm/chart/tsp/{symbol}` - Get test set predictions chart
- `GET /metrics/lstm/chart/tl/{symbol}` - Get training loss chart

### LSTM Sentiment Model Metrics
- `GET /metrics/lstm_sentiment/{symbol}` - Get LSTM sentiment model metrics
- `GET /metrics/lstm_sentiment/chart/tsp/{symbol}` - Get sentiment test predictions chart
- `GET /metrics/lstm_sentiment/chart/tl/{symbol}` - Get sentiment training loss chart

### Predictions
- `GET /predict/lstm?symbol={symbol}&days={days}` - LSTM price prediction
- `GET /predict/lstm_sentiment?symbol={symbol}&days={days}` - LSTM sentiment-based prediction

### Chatbot (Ollama + Gemma 2B)
- `POST /chat` - Chat with a local LLM grounded on your CSVs and optional live prices
    - Body JSON:
        - `prompt` (string, required)
        - `symbols` (string[], optional) - filter to selected tickers
        - `date_start`, `date_end` (YYYY-MM-DD, optional)
        - `use_live_prices` (bool, default true)
        - `model` (string, default `gemma2:2b`)

## 🔧 API Usage Examples

```bash
# Health check
curl http://localhost:8000/

# Get sentiment summary for Apple
curl http://localhost:8000/sentiment_summary/AAPL

# Get LSTM prediction for Apple (30 days of data)
curl "http://localhost:8000/predict/lstm?symbol=AAPL&days=30"

# Get LSTM sentiment prediction for Microsoft
curl "http://localhost:8000/predict/lstm_sentiment?symbol=MSFT&days=60"

# Get model metrics for Google
curl http://localhost:8000/metrics/lstm/GOOGL

# Download sentiment chart for Amazon
curl http://localhost:8000/sentiment_chart/AMZN --output amzn_sentiment.png

# Chatbot: ask about Apple and Microsoft using recent CSVs
curl -X POST http://localhost:8000/chat \
    -H 'Content-Type: application/json' \
    -d '{
        "prompt": "Compare recent performance and predictions for AAPL vs MSFT.",
        "symbols": ["AAPL", "MSFT"],
        "use_live_prices": true
    }'

### Ollama setup (outside the container)
1. Install Ollama: https://ollama.com
2. Start the model once to pull it:
     ```bash
     ollama run gemma2:2b
     ```
3. Ensure the Ollama server is running locally (default http://localhost:11434). If using a remote host or different port, set `OLLAMA_BASE_URL` in the API environment.

Alternatively, run the helper script on Linux:
```bash
bash scripts/install_ollama.sh
```
```

## 🏗️ Project Structure

```
stock-api/
├── app/
│   ├── api.py                      # FastAPI application
│   ├── scheduler.py                # Daily data update scheduler
│   ├── download_datasets.py        # Stock data downloader
│   ├── generate_sentiment.py       # Sentiment analysis generator
│   ├── predict_lstm.py             # LSTM prediction logic
│   ├── predict_lstm_sentiment.py   # LSTM sentiment prediction logic
│   └── save_predictions.py         # Prediction result saver
├── datasets/                       # Stock data CSV files
├── models/
│   ├── lstm/                       # LSTM model files
│   └── lstm_senti/                 # LSTM sentiment model files
├── results/                        # Daily prediction results
├── sentiments/                     # Sentiment analysis results
├── logs/                           # Application logs
├── docker/                         # Docker configuration files
├── requirements.txt                # Python dependencies
├── run.sh                          # Application startup script
└── docker.sh                       # Docker management script
```

## 🤖 Automated Features

### Daily Scheduler
The application includes a background scheduler that runs daily at **3:30 AM UTC** (9:00 AM Sri Lanka time) to:
1. Download latest stock data
2. Generate sentiment analysis
3. Update prediction models
4. Save new predictions

### Manual Script Execution
You can also run individual scripts manually:
```bash
python3 app/download_datasets.py    # Download stock data
python3 app/generate_sentiment.py   # Generate sentiment analysis
python3 app/save_predictions.py     # Save predictions
```

## 🐳 Docker Management

Use the convenience script for easy Docker management:

```bash
./docker.sh build      # Build the Docker image
./docker.sh start      # Start the container
./docker.sh stop       # Stop the container
./docker.sh restart    # Restart the container
./docker.sh logs       # View container logs
./docker.sh shell      # Access container shell
./docker.sh test       # Run tests
```

## 📊 Model Information

### LSTM Models
- **Standard LSTM**: Uses historical price data for predictions
- **LSTM with Sentiment**: Incorporates sentiment analysis for enhanced predictions
- **Training Data**: 60-day lookback window by default
- **Output**: Next day price prediction

### Sentiment Analysis
- **Data Source**: News API integration
- **Processing**: Real-time sentiment scoring
- **Integration**: Combined with LSTM for enhanced predictions
- **Visualization**: Sentiment trend charts

## 🔍 Monitoring and Logs

### View Application Logs
```bash
# Docker logs
./docker.sh logs

# Manual setup logs
tail -f logs/$(date +%Y-%m-%d).log

# Scheduler logs
cat logs/scheduler_log.csv
```

### Health Monitoring
```bash
# Check API health
curl http://localhost:8000/

# Check container status
docker ps

# Check container resources
docker stats stock-api-container
```

## 📋 Requirements

### System Requirements
- Python 3.12+
- 4GB+ RAM (for model training)
- 10GB+ storage space
- Internet connection (for data updates)

### Python Dependencies
See `requirements.txt` for the complete list of dependencies including:
- FastAPI & Uvicorn
- TensorFlow & PyTorch
- Pandas & NumPy
- Scikit-learn
- yfinance
- APScheduler
- Transformers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the logs: `./docker.sh logs`
2. Review the Docker README: `docker/DOCKER_README.md`
3. Open an issue on GitHub

## 🔗 API Keys

The application uses News API for sentiment analysis. API keys are referenced in the code but should be configured as environment variables in production.
---

**Note**: This API is for educational and research purposes. Always consult with financial professionals before making investment decisions.