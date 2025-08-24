from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import os
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, time

import app.scheduler  # ensures the daily scheduler runs

app = FastAPI()

# Test
def get_today():
    now_utc = datetime.utcnow()
    if now_utc.time() < time(2, 45):
        effective_date = (now_utc - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        effective_date = now_utc.strftime('%Y-%m-%d')
    return effective_date

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify your domain)
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/favicon.ico")
def favicon():
    favicon_path = "static/favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/x-icon")
    return JSONResponse(content={"error": "Favicon not found"}, status_code=404)

@app.get("/health")
def health_check():
    return {"status": "ok", "date": get_today()}

@app.get("/")
def root():
    return {
        "message": "Stock Prediction API is running.",
        "created_by": "Sevinda-Herath",
    }


# Sentiments Section

@app.get("/sentiment_summary/{symbol}")
def get_summary(symbol: str):
    file_path = f"sentiments/summary/{get_today()}/{symbol.upper()}_summary.csv"
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "Summary not found"}, status_code=404)
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")[0]

@app.get("/sentiment_chart/{symbol}")
def get_chart(symbol: str):
    chart_path = f"sentiments/charts/{get_today()}/{symbol.upper()}_chart.png"
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="image/png")
    return JSONResponse(content={"error": "Chart not found"}, status_code=404)

# Metrics Section

# LSTM
@app.get("/metrics/lstm/{symbol}")
def get_lstm_metrics(symbol: str):
    file_path = f"model-metrics-charts/lstm/metrics/{symbol.upper()}_lstm_model_metrics.csv"
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "Metrics not found"}, status_code=404)
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")[0]

@app.get("/metrics/lstm/chart/tsp/{symbol}")
def get_chart(symbol: str):
    chart_path = f"model-metrics-charts/lstm/test_set_predictions/{symbol.upper()}_lstm_test_plot.png"
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="image/png")
    return JSONResponse(content={"error": "Chart not found"}, status_code=404)

@app.get("/metrics/lstm/chart/tl/{symbol}")
def get_chart(symbol: str):
    chart_path = f"model-metrics-charts/lstm/training_loss/{symbol.upper()}_lstm_loss_plot.png"
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="image/png")
    return JSONResponse(content={"error": "Chart not found"}, status_code=404)

# LSTM Sentiment
@app.get("/metrics/lstm_sentiment/{symbol}")
def get_lstm_sentiment_metrics(symbol: str):
    file_path = f"model-metrics-charts/lstm_senti/metrics/{symbol.upper()}_lstm_senti_model_metrics.csv"
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "Metrics not found"}, status_code=404)
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")[0]

@app.get("/metrics/lstm_sentiment/chart/tsp/{symbol}")
def get_chart(symbol: str):
    chart_path = f"model-metrics-charts/lstm_senti/test_set_predictions/{symbol.upper()}_lstm_senti_test_plot.png"
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="image/png")
    return JSONResponse(content={"error": "Chart not found"}, status_code=404)

@app.get("/metrics/lstm_sentiment/chart/tl/{symbol}")
def get_chart(symbol: str):
    chart_path = f"model-metrics-charts/lstm_senti/training_loss/{symbol.upper()}_lstm_senti_loss_plot.png"
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="image/png")
    return JSONResponse(content={"error": "Chart not found"}, status_code=404)

# Prediction Section

@app.get("/predict/lstm")
def predict_price(symbol: str = Query(...), days: int = Query(60)):
    from app.predict_lstm import predict_lstm_price
    try:
        price = predict_lstm_price(symbol.upper(), days)
        return {
            "date": get_today(),
            "stock": symbol.upper(),
            "predicted_price_for_tommorow": float(round(price, 2))  # Fix: convert numpy.float32 to float
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.get("/predict/lstm_sentiment")
def predict_price_sentiment(symbol: str = Query(...), days: int = Query(60)):
    from app.predict_lstm_sentiment import predict_lstm_sentiment_price
    try:
        price = predict_lstm_sentiment_price(symbol.upper(), days)
        return {
            "date": get_today(),
            "stock": symbol.upper(),
            "predicted_price_for_tommorow": float(round(price, 2))  # Fix: convert numpy.float32 to float
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
