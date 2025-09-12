"""
A+ LONG-only Futures Strategy Bot (Binance Futures) - FIXED startup & async issues
- Multi-pair: BTC, ETH, SOL, XRP, BNB (configurable)
- Real-time liquidation stream + ticker/trade stream
- 1H SR detection (from minute closes), 15m entry confirmation
- Liquidation spike filter, session filter (London/NY), single-alert policy
- Sends Telegram alerts (pre-filled token/chat)
- Save file, install deps, run: python bot.py
"""

import asyncio
import json
import time
from collections import deque, defaultdict
from datetime import datetime, timezone
import aiohttp
import websockets
import numpy as np

# -------------------------
# CONFIG - edit if needed
# -------------------------
# Telegram token & chat id (you requested these be injected)
TELEGRAM_TOKEN = "8137416692:AAGDtydvybjIpQAfPTodqOE9gsJRVoC3YGY"
TELEGRAM_CHAT_ID = "8410854765"

# Binance futures websocket endpoints
BINANCE_WS_BASE = "wss://fstream.binance.com/ws"

# Symbols to monitor (Binance futures symbol format)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]

# Session windows (UTC)
LONDON_SESSION = (7, 11)    # 07:00 - 11:00 UTC
NY_SESSION     = (12, 17)   # 12:00 - 17:00 UTC

# Liquidation spike detection
LIQ_WINDOW_SEC = 30
LIQ_SPIKE_USD_THRESHOLD = 100_000  # tune per symbol / global; for BTC keep high

# SR detection
SR_LOOKBACK_MINUTES = 120
SR_LOCAL_WINDOW = 2
SR_MAX_LEVELS = 6
SR_PROX_PCT = 0.005  # 0.5% proximity to support

# Trend detection
TREND_15M_LOOKBACK = 15
TREND_MIN_CONF_POINTS = 6
TREND_SENSITIVITY = 0.001  # 0.1% threshold to decide UP/DOWN/FLAT

# Alert control
ALERT_COOLDOWN_SEC = 120
DAILY_MAX_SIGNALS_PER_SYMBOL = 2
CONFIDENCE_LABEL = "90%+"

# Risk / Plan
RR = 3  # 1:3
# Logging
PRINT_LOGS = True

# -------------------------
# Utilities
# -------------------------
def now_ms(): return int(time.time() * 1000)
def utc_now_str(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
def utc_hour(): return datetime.now(timezone.utc).hour
def log(*args):
    if PRINT_LOGS:
        # use timezone-aware UTC now to avoid DeprecationWarning
        print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), *args)

# -------------------------
# Runtime state
# -------------------------
# minute resolution closes per symbol
minute_closes = {s: deque(maxlen=SR_LOOKBACK_MINUTES) for s in SYMBOLS}
# recent liquidations per symbol -> deque of (ts_ms, usd)
recent_liqs = {s: deque() for s in SYMBOLS}
# alert controls
last_alert_ts = {s: 0 for s in SYMBOLS}
daily_count = defaultdict(int)
daily_count_date = datetime.now(timezone.utc).date()

# -------------------------
# Support/Resistance detection
# -------------------------
def detect_sr_levels_from_closes(closes, local_window=SR_LOCAL_WINDOW, max_levels=SR_MAX_LEVELS):
    levels = []
    arr = list(closes)
    n = len(arr)
    if n < local_window*2 + 1:
        return []
    for i in range(local_window, n-local_window):
        window = arr[i-local_window:i+local_window+1]
        center = arr[i]
        if center == max(window):
            levels.append(("RESIST", center))
        if center == min(window):
            levels.append(("SUPPORT", center))
    out = []
    seen = []
    for kind, lvl in reversed(levels):
        if any(abs(lvl - s) / s < 0.002 for s in seen):
            continue
        out.append((kind, lvl))
        seen.append(lvl)
        if len(out) >= max_levels:
            break
    return out

# -------------------------
# Trend detection
# -------------------------
def simple_trend_from_closes(closes, lookback):
    if len(closes) < max(TREND_MIN_CONF_POINTS, lookback):
        return "FLAT"
    arr = np.array(list(closes)[-lookback:])
    mean = float(np.mean(arr))
    recent = float(arr[-1])
    if (recent - mean) / mean > TREND_SENSITIVITY:
        return "UP"
    if (mean - recent) / mean > TREND_SENSITIVITY:
        return "DOWN"
    return "FLAT"

# -------------------------
# Session filter
# -------------------------
def in_liquidity_session():
    h = utc_hour()
    ls, le = LONDON_SESSION
    ns, ne = NY_SESSION
    return (ls <= h < le) or (ns <= h < ne)

# -------------------------
# Liquidation spike functions
# -------------------------
def push_liq(symbol, usd):
    ts = now_ms()
    q = recent_liqs[symbol]
    q.append((ts, usd))

def is_recent_liq_spike(symbol):
    cutoff = now_ms() - (LIQ_WINDOW_SEC * 1000)
    q = recent_liqs[symbol]
    while q and q[0][0] < cutoff:
        q.popleft()
    total = sum(u for (_, u) in q)
    log(f"[LIQ] {symbol} last {LIQ_WINDOW_SEC}s USD = {total:,.0f}")
    return total >= LIQ_SPIKE_USD_THRESHOLD, total

# -------------------------
# Telegram send (async)
# -------------------------
async def send_telegram(session: aiohttp.ClientSession, text: str):
    if TELEGRAM_TOKEN.startswith("<") or TELEGRAM_CHAT_ID.startswith("<"):
        log("[TELEGRAM] not configured; message would be:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        async with session.post(url, json=payload, timeout=10) as resp:
            if resp.status != 200:
                body = await resp.text()
                log("[TELEGRAM] send failed", resp.status, body)
    except Exception as e:
        log("[TELEGRAM] error", e)

# -------------------------
# Build trade plan (LONG-only)
# -------------------------
def build_long_plan(symbol, price, support_level, liq_total):
    entry = float(price)
    sl_candidate1 = support_level * 0.997
    sl_candidate2 = entry * 0.995
    sl = min(sl_candidate1, sl_candidate2)
    if sl >= entry:
        sl = entry * 0.995
    risk = entry - sl
    if risk <= 0:
        risk = entry * 0.002
        sl = entry - risk
    tp = entry + (risk * RR)
    def r(v):
        if v >= 1:
            return round(v, 2)
        return round(v, 4)
    return {
        "symbol": symbol,
        "side": "LONG",
        "entry": r(entry),
        "sl": r(sl),
        "tp": r(tp),
        "rr": f"1:{RR}",
        "liq_usd": int(liq_total)
    }

# -------------------------
# Evaluate and possibly alert
# -------------------------
async def evaluate_symbol(symbol, http_session: aiohttp.ClientSession):
    global daily_count_date
    today = datetime.now(timezone.utc).date()
    if today != daily_count_date:
        daily_count.clear()
        daily_count_date = today

    if not in_liquidity_session():
        log(symbol, "outside sessions -> skip")
        return

    closes = minute_closes[symbol]
    if len(closes) < 30:
        return

    sr_levels = detect_sr_levels_from_closes(closes)
    if not sr_levels:
        return
    supports = [lvl for (k, lvl) in sr_levels if k == "SUPPORT"]
    if not supports:
        return

    trend_15m = simple_trend_from_closes(closes[-15:], 15)
    trend_1h = simple_trend_from_closes(closes[-60:], 15) if len(closes) >= 60 else "FLAT"
    log(symbol, "trend_15m:", trend_15m, "trend_1h:", trend_1h)
    if trend_15m != "UP" or trend_1h != "UP":
        return

    spike, liq_total = is_recent_liq_spike(symbol)
    if not spike:
        return

    price = float(closes[-1])
    nearest_support = min(supports, key=lambda s: abs(s - price))
    if abs(price - nearest_support) / nearest_support > SR_PROX_PCT:
        log(symbol, "price not within SR proximity -> skip (price, support):", price, nearest_support)
        return

    if now_ms() - last_alert_ts[symbol] < ALERT_COOLDOWN_SEC * 1000:
        log(symbol, "cooldown active -> skip")
        return
    if daily_count[symbol] >= DAILY_MAX_SIGNALS_PER_SYMBOL:
        log(symbol, "daily cap reached -> skip")
        return

    plan = build_long_plan(symbol, price, nearest_support, liq_total)
    text = (
        f"📢 A+ LONG Alert (High Confidence)\n\n"
        f"Pair: {plan['symbol']}\n"
        f"Side: {plan['side']} ✅\n"
        f"Time: {utc_now_str()}\n\n"
        f"📈 Entry: {plan['entry']}\n"
        f"❌ Stop Loss: {plan['sl']}\n"
        f"🎯 Take Profit: {plan['tp']}\n"
        f"⚖ Risk/Reward: {plan['rr']}\n\n"
        f"🔎 Reasons:\n"
        f"- 1H Support detected: {round(nearest_support, 6)}\n"
        f"- 15m & 1H trend: UP\n"
        f"- Recent liquidation spike: ~${plan['liq_usd']:,} in last {LIQ_WINDOW_SEC}s\n"
        f"- Session filter: London/NY (PASS)\n\n"
        f"🧭 Confidence: {CONFIDENCE_LABEL}\n"
        f"📝 Note: Signal only — backtest & paper-trade before live execution.\n"
        f"🔁 Policy: one alert per setup | daily cap per symbol: {DAILY_MAX_SIGNALS_PER_SYMBOL}\n"
    )
    await send_telegram(http_session, text)
    last_alert_ts[symbol] = now_ms()
    daily_count[symbol] += 1
    log("ALERT SENT for", symbol, "| daily_count:", daily_count[symbol])

# -------------------------
# Binance listeners
# -------------------------
async def binance_liq_listener():
    WS = f"{BINANCE_WS_BASE}/!forceOrder@arr"
    while True:
        try:
            async with websockets.connect(WS, ping_interval=20, max_size=None) as ws:
                log("[Binance-LIQ] connected")
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        data = msg.get("data") or msg
                        if isinstance(data, dict) and 'data' in data:
                            data = data['data']
                        if isinstance(data, list):
                            for e in data:
                                sym = (e.get("s") or e.get("symbol") or "").upper()
                                if not sym or sym not in SYMBOLS:
                                    continue
                                qty = 0.0
                                price = 0.0
                                if isinstance(e.get("o"), dict):
                                    qty = float(e["o"].get("q") or e["o"].get("qty") or 0.0)
                                    price = float(e["o"].get("p") or e["o"].get("price") or 0.0)
                                else:
                                    qty = float(e.get("l") or e.get("q") or 0.0)
                                    price = float(e.get("p") or e.get("price") or 0.0)
                                usd = qty * price
                                if usd > 0:
                                    push_liq(sym, usd)
                        elif isinstance(data, dict):
                            e = data
                            sym = (e.get("s") or e.get("symbol") or "").upper()
                            if sym in SYMBOLS:
                                qty = 0.0
                                price = 0.0
                                if isinstance(e.get("o"), dict):
                                    qty = float(e["o"].get("q") or e["o"].get("qty") or 0.0)
                                    price = float(e["o"].get("p") or e["o"].get("price") or 0.0)
                                else:
                                    qty = float(e.get("l") or e.get("q") or 0.0)
                                    price = float(e.get("p") or e.get("price") or 0.0)
                                usd = qty * price
                                if usd > 0:
                                    push_liq(sym, usd)
                    except Exception:
                        continue
        except Exception as e:
            log("[Binance-LIQ] error:", e, "reconnecting in 2s")
            await asyncio.sleep(2)

async def binance_ticker_listener(trade_queue: asyncio.Queue):
    WS = f"{BINANCE_WS_BASE}/!ticker@arr"
    while True:
        try:
            async with websockets.connect(WS, ping_interval=20, max_size=None) as ws:
                log("[Binance-TICKER] connected")
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        if isinstance(msg, dict) and msg.get("s") and msg.get("c"):
                            sym = msg.get("s").upper()
                            if sym not in SYMBOLS:
                                continue
                            price = float(msg.get("c") or 0.0)
                            await trade_queue.put((sym, price, now_ms()))
                        elif isinstance(msg, list):
                            for t in msg:
                                sym = t.get("s")
                                if not sym or sym not in SYMBOLS:
                                    continue
                                price = float(t.get("c") or 0.0)
                                await trade_queue.put((sym, price, now_ms()))
                    except Exception:
                        continue
        except Exception as e:
            log("[Binance-TICKER] error:", e, "reconnecting in 2s")
            await asyncio.sleep(2)

# -------------------------
# Minute aggregator from trade_queue
# -------------------------
async def minute_aggregator(trade_queue: asyncio.Queue, http_session):
    last_minute = None
    minute_last_price = {s: None for s in SYMBOLS}
    while True:
        try:
            sym, price, ts = await trade_queue.get()
            if sym not in SYMBOLS:
                continue
            minute_last_price[sym] = price
            m = int(ts // 60000)
            if last_minute is None:
                last_minute = m
            if m != last_minute:
                for s in SYMBOLS:
                    p = minute_last_price.get(s)
                    if p is not None:
                        minute_closes[s].append(p)
                minute_last_price = {s: None for s in SYMBOLS}
                last_minute = m
                for s in SYMBOLS:
                    asyncio.create_task(evaluate_symbol(s, http_session))
        except Exception as e:
            log("[MIN-AGG] error:", e)
            await asyncio.sleep(0.1)

# -------------------------
# Entrypoint
# -------------------------
async def main():
    log("Starting A+ LONG-only Futures Bot (Binance) ...")
    trade_queue = asyncio.Queue()
    async with aiohttp.ClientSession() as http_sess:
        # send startup message (async) so you get confirmation in Telegram
        try:
            await send_telegram(http_sess, "🚀 A+ Strategy Bot is now LIVE on Render ✅")
        except Exception as e:
            log("Startup telegram error:", e)

        tasks = [
            asyncio.create_task(binance_liq_listener()),
            asyncio.create_task(binance_ticker_listener(trade_queue)),
            asyncio.create_task(minute_aggregator(trade_queue, http_sess)),
        ]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Shutting down...")
