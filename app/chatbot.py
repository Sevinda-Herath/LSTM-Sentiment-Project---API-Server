import os
import json
import requests
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta, time
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None  # yfinance may not be available in minimal environments


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PREDICTIONS_START = "2025-07-14"


def _effective_today() -> str:
    now_utc = datetime.utcnow()
    if now_utc.time() < time(2, 45):
        return (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
    return now_utc.strftime("%Y-%m-%d")


def _latest_date_dir(base_dir: str) -> Optional[str]:
    """Return latest YYYY-MM-DD subfolder path under base_dir, or None."""
    if not os.path.isdir(base_dir):
        return None
    dates = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            try:
                # Validate date format
                datetime.strptime(name, "%Y-%m-%d")
                dates.append(name)
            except ValueError:
                continue
    if not dates:
        return None
    latest = sorted(dates)[-1]
    return os.path.join(base_dir, latest)


def _safe_read_csv(path: str, usecols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, usecols=usecols)
    except Exception:
        return None


def _fmt_money(x: Optional[float]) -> str:
    return "N/A" if x is None else f"${x:,.2f}"


def _fetch_live_price(symbol: str) -> Optional[Tuple[float, str]]:
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="1d", interval="1m")
        if df is not None and not df.empty and "Close" in df:
            price = float(df["Close"].iloc[-1])
            ts = df.index[-1]
            # Format timestamp in ISO without microseconds
            ts_str = ts.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, 'tz_convert') else str(ts)
            return price, ts_str
    except Exception:
        return None
    return None


def _summarize_series(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Tuple[str, Optional[float], Optional[pd.Timestamp]]:
    if df is None or df.empty:
        return "No data available.", None, None
    sub = df.copy()
    if start is not None:
        sub = sub[sub["Date"] >= start]
    if end is not None:
        sub = sub[sub["Date"] <= end]
    if sub.empty:
        return "No rows in selected date range.", None, None
    sub = sub.sort_values("Date")
    start_close = float(sub["Close"].iloc[0])
    end_close = float(sub["Close"].iloc[-1])
    end_date = sub["Date"].iloc[-1]
    change = end_close - start_close
    pct = (change / start_close * 100.0) if start_close else 0.0
    line = (
        f"Range: {sub['Date'].iloc[0].date()} → {sub['Date'].iloc[-1].date()} | "
        f"Start: {_fmt_money(start_close)} | End: {_fmt_money(end_close)} | "
        f"Δ: {_fmt_money(change)} ({pct:+.2f}%)"
    )
    return line, end_close, end_date


def build_context(
    symbols: List[str],
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    max_rows: int = 120,
    use_live_prices: bool = True,
) -> Tuple[str, Dict[str, str]]:
    """Build a concise textual context from CSVs and optional live prices.

    Returns: (context_text, meta)
    meta contains reference dates for results/sentiments.
    """
    # Determine effective end (yesterday) and clamp to 30-day window
    eff_today_str = _effective_today()
    eff_end_default = pd.to_datetime(eff_today_str)
    req_start = pd.to_datetime(date_start) if date_start else None
    req_end = pd.to_datetime(date_end) if date_end else None

    # Used end is min(requested end, effective today)
    used_end = min(req_end, eff_end_default) if req_end is not None else eff_end_default
    # Used start: if requested start is provided, cap to within 30 days before used_end
    thirty_days_before = used_end - timedelta(days=30)
    if req_start is not None and req_start > thirty_days_before:
        used_start = req_start
    else:
        used_start = thirty_days_before

    results_dir = _latest_date_dir("results")
    sentiments_summary_dir = _latest_date_dir(os.path.join("sentiments", "summary"))

    # Load daily predictions if present
    lstm_df = _safe_read_csv(os.path.join(results_dir, "lstm.csv")) if results_dir else None
    lstm_senti_df = _safe_read_csv(os.path.join(results_dir, "lstm_senti.csv")) if results_dir else None

    lines: List[str] = []
    lines.append("Context compiled from local CSVs (up to yesterday) and optional live prices.\n")
    lines.append(f"Notes: Predictions available since {PREDICTIONS_START}. Earlier periods won't have prediction values.\n")

    for sym in symbols:
        lines.append(f"=== {sym} ===")

        # Market data
        ds_path = os.path.join("datasets", f"{sym}_daily_data.csv")
        ds = _safe_read_csv(ds_path)
        if ds is not None and not ds.empty and "Date" in ds.columns and "Close" in ds.columns:
            ds["Date"] = pd.to_datetime(ds["Date"])
            # keep at most max_rows rows for brevity
            ds = ds.sort_values("Date").tail(max_rows)
            summary_line, last_close, last_date = _summarize_series(ds, used_start, used_end)
            lines.append(f"Market summary: {summary_line}")
        else:
            last_close = None
            last_date = None
            lines.append("Market summary: Not available.")

        # Predictions
        pred_parts = []
        try:
            if lstm_df is not None and not lstm_df.empty:
                row = lstm_df.loc[lstm_df["symbol"] == sym]
                if not row.empty and "predicted_price" in row:
                    pred_price = float(row['predicted_price'].iloc[0])
                    pred_date = str(row['date'].iloc[0]) if 'date' in row else ''
                    pred_parts.append(f"LSTM: {_fmt_money(pred_price)} (as of {pred_date}, predicts next day)")
        except Exception:
            pass
        try:
            if lstm_senti_df is not None and not lstm_senti_df.empty:
                row = lstm_senti_df.loc[lstm_senti_df["symbol"] == sym]
                if not row.empty and "predicted_price" in row:
                    pred_price = float(row['predicted_price'].iloc[0])
                    pred_date = str(row['date'].iloc[0]) if 'date' in row else ''
                    pred_parts.append(f"LSTM+Sentiment: {_fmt_money(pred_price)} (as of {pred_date}, predicts next day)")
        except Exception:
            pass
        if pred_parts:
            lines.append("Predictions: " + " | ".join(pred_parts))
        else:
            lines.append("Predictions: Not available.")

        # Sentiment summary
        senti_line = "Not available."
        if sentiments_summary_dir:
            senti_path = os.path.join(sentiments_summary_dir, f"{sym}_summary.csv")
            sdf = _safe_read_csv(senti_path)
            if sdf is not None and not sdf.empty:
                try:
                    rec = sdf.iloc[0].to_dict()
                    senti_line = (
                        f"Articles: {int(rec.get('total_articles', 0))} | "
                        f"Pos: {int(rec.get('positive_count', 0))}, Neu: {int(rec.get('neutral_count', 0))}, Neg: {int(rec.get('negative_count', 0))}"
                    )
                except Exception:
                    pass
        lines.append("Sentiment: " + senti_line)

        # Live price (optional)
        live_line = ""
        if use_live_prices:
            live = _fetch_live_price(sym)
            if live is not None:
                live_price, live_ts = live
                live_line = f"Live price: {_fmt_money(live_price)} (as of {live_ts})"
            elif last_close is not None:
                last_date_str = last_date.strftime('%Y-%m-%d') if last_date is not None else 'N/A'
                live_line = f"Live price: Unavailable. Last close: {_fmt_money(last_close)} ({last_date_str})"
            else:
                live_line = "Live price: Unavailable."
            lines.append(live_line)

        lines.append("")

    meta = {
        "results_date": os.path.basename(results_dir) if results_dir else "",
        "sentiments_date": os.path.basename(sentiments_summary_dir) if sentiments_summary_dir else "",
        "used_start_date": used_start.strftime('%Y-%m-%d') if isinstance(used_start, pd.Timestamp) else "",
        "used_end_date": used_end.strftime('%Y-%m-%d') if isinstance(used_end, pd.Timestamp) else "",
    }

    return "\n".join(lines), meta


def call_ollama_chat(messages: List[Dict[str, str]], model: str = "gemma2:2b", stream: bool = False) -> str:
    """Call Ollama's /api/chat endpoint and return assistant text content.

    Expects local Ollama server running (e.g., `ollama run gemma2:2b`).
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Non-streaming returns a single object with message
        if isinstance(data, dict) and "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "")
        # Some variants may return chunks; coalesce if list
        if isinstance(data, list):
            parts = []
            for chunk in data:
                msg = chunk.get("message", {}) if isinstance(chunk, dict) else {}
                parts.append(msg.get("content", ""))
            return "".join(parts)
        return ""
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to reach Ollama at {url}: {e}")
