from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import os
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, time

import app.scheduler  # ensures the daily scheduler runs
from pydantic import BaseModel, Field
from typing import List, Optional

from app.chatbot import build_context, call_ollama_chat

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
    allow_methods=["GET", "POST", "OPTIONS"],
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


# Chatbot Section

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User question or instruction")
    symbols: List[str] = Field(default_factory=list, description="Stock symbols to include in context")
    date_start: Optional[str] = Field(None, description="YYYY-MM-DD inclusive start date for data slice")
    date_end: Optional[str] = Field(None, description="YYYY-MM-DD inclusive end date for data slice")
    use_live_prices: bool = Field(True, description="Whether to include live prices when available")
    model: str = Field("gemma2:2b", description="Ollama model name, e.g., gemma2:2b")


class ChatResponse(BaseModel):
    answer: str
    context_meta: dict


@app.post("/chat", response_model=ChatResponse)
def chat_with_csv(req: ChatRequest = Body(...)):
    try:
        # Auto-detect symbols from datasets if not provided
        symbols = [s.upper() for s in (req.symbols or [])]
        if not symbols:
            try:
                files = os.listdir("datasets")
                symbols = [
                    f.replace("_daily_data.csv", "")
                    for f in files if f.endswith("_daily_data.csv")
                ]
            except Exception:
                symbols = []

        # Build context from local CSVs and optional live prices
        context_text, meta = build_context(
            symbols=symbols,
            date_start=req.date_start,
            date_end=req.date_end,
            use_live_prices=req.use_live_prices,
        )

        # Compute current dates
        now_utc_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        effective_today = get_today()

        # Build a user-facing period label
        if req.date_start and req.date_end:
            period_label = f"{req.date_start} to {req.date_end}"
        elif req.date_start and not req.date_end:
            period_label = f"{req.date_start} to {effective_today}"
        elif not req.date_start and req.date_end:
            period_label = f"up to {req.date_end}"
        else:
            period_label = "recent window (~last 120 trading days)"

        system_prompt = (
            f"You are a concise financial data assistant. Today's date/time is {now_utc_str}. "
            f"Use the effective data date {effective_today} for daily CSV outputs when relevant."
            "Always state the date or the time period selected by the user before starting the conversation"
            "Use only the provided context, the user's prompt, and general market reasoning. "
            "If data is missing, say so. Prefer concrete numbers and short explanations. "
            f"Important: Start your answer with this exact first line: Period: {period_label}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nUser question: {req.prompt}"},
        ]

        answer = call_ollama_chat(messages, model=req.model, stream=False)

        # Enrich meta with date info
        meta = dict(meta or {})
        meta.update({
            "now_utc": now_utc_str,
            "effective_date": effective_today,
            "period": period_label,
        })

        return ChatResponse(answer=answer.strip(), context_meta=meta)
    except ConnectionError as ce:
        return JSONResponse(status_code=503, content={"error": str(ce)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
