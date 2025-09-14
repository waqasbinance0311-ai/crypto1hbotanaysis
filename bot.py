"""
Final deploy-ready bot.py
- Multi-symbol Binance Futures watcher
- 15m + 1h confluence SR detection
- Liquidation spike filter
- Session filter (London/NY)
- BUY/SELL with fixed SL/TP (2.5 / 8.5)
- Robust deque->list handling (no slice TypeError)
- Async Telegram alerts
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
# CONFIG (edit per your needs)
# -------------------------
TELEGRAM_TOKEN = "8137416692:AAGDtydvybjIpQAfPTodqOE9gsJRVoC3YGY"   # <-- replace with your token
TELEGRAM_CHAT_ID = "8410854765"                                     # <-- replace with your chat id

BINANCE_WS_BASE = "wss://fstream.binance.com/ws"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]

# Sessions in UTC (hours)
LONDON_SESSION = (7, 11)    # 07:00 - 11:00 UTC
NY_SESSION     = (12, 17)   # 12:00 - 17:00 UTC

# Liquidation spike detection
LIQ_WINDOW_SEC = 30
# Global thresholds; tune per symbol (BTC >> altcoins)
LIQ_SPIKE_USD_THRESHOLD = {
    "BTCUSDT": 200_000,
    "ETHUSDT": 80_000,
    "SOLUSDT": 20_000,
    "XRPUSDT": 10_000,
    "BNBUSDT": 15_000
}
LIQ_STRONG_MULT = 3  # strong spike multiplier for NY allowance

# SR detection & confluence
SR_LOOKBACK_MINUTES = 240    # keep 4 hours of minute closes by default
SR_LOCAL_WINDOW = 2
SR_MAX_LEVELS = 8
SR_PROX_PCT = 0.005         # 0.5% proximity to SR for being 'near'
CONFLUENCE_PCT = 0.002      # 0.2% tolerance between 15m and 1h levels

# Trend detection
TREND_MIN_CONF_POINTS = 6
TREND_SENSITIVITY = 0.001   # 0.1% threshold

# Alerts control
ALERT_COOLDOWN_SEC = 120
DAILY_MAX_SIGNALS_PER_SYMBOL = 2
CONFIDENCE_LABEL = "High"

# Risk: fixed SL/TP
FIXED_SL = 2.5
FIXED_TP = 8.5

# Logging & test
PRINT_LOGS = True
TEST_MODE = False   # set True to bypass session checks for testing locally

# -------------------------
# Utilities
# -------------------------
def now_ms(): return int(time.time() * 1000)
def utc_now_str(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
def utc_hour(): return datetime.now(timezone.utc).hour
def log(*args):
    if PRINT_LOGS:
        print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), *args)

# -------------------------
# Runtime state
# -------------------------
minute_closes = {s: deque(maxlen=SR_LOOKBACK_MINUTES) for s in SYMBOLS}
recent_liqs = {s: deque() for s in SYMBOLS}   # (ts_ms, usd)
last_alert_ts = {s: 0 for s in SYMBOLS}
daily_count = defaultdict(int)
daily_count_date = datetime.now(timezone.utc).date()

# -------------------------
# Helpers: time/session
# -------------------------
def in_liquidity_session():
    if TEST_MODE:
        return True
    h = utc_hour()
    ls, le = LONDON_SESSION
    ns, ne = NY_SESSION
    return (ls <= h < le) or (ns <= h < ne)

def is_ny_session():
    if TEST_MODE:
        return False
    h = utc_hour()
    ns, ne = NY_SESSION
    return ns <= h < ne

# -------------------------
# Helpers: liquidations
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
    # get threshold per symbol or fallback
    th = LIQ_SPIKE_USD_THRESHOLD.get(symbol, 50_000)
    log(f"[LIQ] {symbol} last {LIQ_WINDOW_SEC}s USD = {total:,.0f} (th={th:,})")
    return total >= th, total

# -------------------------
# Helpers: resample minute closes into TF closes
# -------------------------
def resample_closes_to_tf(minute_list, tf_minutes):
    """
    minute_list: list of minute closes (oldest->newest)
    produce aggregated TF closes by taking last close of each tf_minutes bucket
    """
    if not minute_list:
        return []
    n = len(minute_list)
    # align from the end - produce groups of size tf_minutes starting from the newest backwards
    out = []
    # create groups from start so result is oldest->newest
    # simple approach: take every tf_minutes-th element counting from start with possible leading smaller group
    remainder = n % tf_minutes
    start = 0 if remainder == 0 else remainder
    for i in range(start, n, tf_minutes):
        group = minute_list[i:i+tf_minutes]
        if group:
            out.append(group[-1])
    return out

# -------------------------
# SR / pivot detection
# -------------------------
def detect_sr_levels_from_closes(closes, local_window=SR_LOCAL_WINDOW, max_levels=SR_MAX_LEVELS):
    """Return list of (kind, level) newest-first up to max_levels."""
    arr = list(closes)
    n = len(arr)
    if n < local_window*2 + 1:
        return []
    levels = []
    for i in range(local_window, n - local_window):
        window = arr[i-local_window:i+local_window+1]
        center = arr[i]
        if center == max(window):
            levels.append(("RESIST", center))
        if center == min(window):
            levels.append(("SUPPORT", center))
    out = []
    seen = []
    for kind, lvl in reversed(levels):  # most recent pivots first
        if any(abs(lvl - s) / s < 0.002 for s in seen):
            continue
        out.append((kind, lvl))
        seen.append(lvl)
        if len(out) >= max_levels:
            break
    return out

# -------------------------
# Confluence detection (15m vs 1h)
# -------------------------
def get_confluence_levels_from_minute_closes(minute_list):
    closes_15 = resample_closes_to_tf(minute_list, 15)
    closes_60 = resample_closes_to_tf(minute_list, 60)
    if not closes_15 or not closes_60:
        return []
    sr15 = detect_sr_levels_from_closes(closes_15, local_window=1, max_levels=12)
    sr60 = detect_sr_levels_from_closes(closes_60, local_window=1, max_levels=12)
    confl = []
    for k15, l15 in sr15:
        for k60, l60 in sr60:
            avg = (l15 + l60) / 2.0
            if k15 == k60 and abs(l15 - l60) / avg <= CONFLUENCE_PCT:
                confl.append((k15, avg))
    # dedupe by proximity
    out = []
    seen = []
    for k, lvl in confl:
        if any(abs(lvl - s) / s < 0.002 for s in seen):
            continue
        out.append((k, lvl))
        seen.append(lvl)
    return out

# -------------------------
# Simple trend detection
# -------------------------
def simple_trend_from_closes(closes, lookback):
    arr = list(closes)
    if len(arr) < max(TREND_MIN_CONF_POINTS, lookback):
        return "FLAT"
    a = np.array(arr[-lookback:], dtype=float)
    mean = float(np.mean(a))
    recent = float(a[-1])
    if (recent - mean) / mean > TREND_SENSITIVITY:
        return "UP"
    if (mean - recent) / mean > TREND_SENSITIVITY:
        return "DOWN"
    return "FLAT"

# -------------------------
# Telegram
# -------------------------
async def send_telegram(session: aiohttp.ClientSession, text: str):
    if TELEGRAM_TOKEN.startswith("<") or TELEGRAM_CHAT_ID.startswith("<"):
        log("[TELEGRAM] not configured; would send:", text)
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
# Build trade plan (LONG/SHORT fixed SL/TP)
# -------------------------
def build_trade_plan(symbol, price, side, liq_total):
    entry = float(price)
    if side in ("LONG", "BUY", "BUY_LONG"):
        sl = entry - FIXED_SL
        tp = entry + FIXED_TP
        side_label = "LONG"
    else:
        sl = entry + FIXED_SL
        tp = entry - FIXED_TP
        side_label = "SHORT"

    def r(v):
        if abs(v) >= 1:
            return round(v, 2)
        return round(v, 4)

    rr_val = None
    try:
        rr_val = round(abs(tp - entry) / abs(entry - sl), 2) if (entry - sl) != 0 else None
    except Exception:
        rr_val = None
    rr_str = f"1:{int(rr_val)}" if rr_val and rr_val >= 1 else (f"{rr_val}" if rr_val else "N/A")

    return {
        "symbol": symbol,
        "side": side_label,
        "entry": r(entry),
        "sl": r(sl),
        "tp": r(tp),
        "rr": rr_str,
        "liq_usd": int(liq_total)
    }

# -------------------------
# Core evaluation logic (confluence + TOD + liq + trend)
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

    closes_deque = minute_closes[symbol]
    closes = list(closes_deque)        # convert to list to allow slicing safely
    if len(closes) < 30:
        # not enough minute data yet
        return

    # compute SRs and confluence
    sr_15m = detect_sr_levels_from_closes(resample_closes_to_tf(closes, 15), local_window=1, max_levels=8)
    sr_1h  = detect_sr_levels_from_closes(resample_closes_to_tf(closes, 60), local_window=1, max_levels=8)
    confluence_levels = get_confluence_levels_from_minute_closes(closes)

    # liquidation check
    spike, liq_total = is_recent_liq_spike(symbol)
    strong_spike = spike and liq_total >= (LIQ_SPIKE_USD_THRESHOLD.get(symbol, 50_000) * LIQ_STRONG_MULT)

    log(symbol, "SR15:", sr_15m, "SR1H:", sr_1h, "CONFL:", confluence_levels, "LIQ:", liq_total)

    # require confluence OR allow NY + very-strong spike
    if not confluence_levels and not (is_ny_session() and strong_spike):
        return

    # aggregated closes for trend checks
    closes_15 = resample_closes_to_tf(closes, 15)
    closes_60 = resample_closes_to_tf(closes, 60)
    trend_15m = simple_trend_from_closes(closes_15, 15) if len(closes_15) >= 15 else "FLAT"
    trend_1h  = simple_trend_from_closes(closes_60, 15) if len(closes_60) >= 15 else "FLAT"
    log(symbol, "trend_15m:", trend_15m, "trend_1h:", trend_1h)

    price = float(closes[-1])

    # decide side by confluence & trend
    side = None
    nearest = None
    for kind, lvl in confluence_levels:
        if abs(price - lvl) / lvl <= SR_PROX_PCT:
            if kind == "SUPPORT" and trend_15m == "UP" and trend_1h == "UP":
                side = "LONG"; nearest = lvl; break
            if kind == "RESIST" and trend_15m == "DOWN" and trend_1h == "DOWN":
                side = "SHORT"; nearest = lvl; break

    # if no confluence triggered but NY-exception triggered earlier, try 15m SR near price
    if not side and (is_ny_session() and strong_spike):
        supports = [lvl for (k, lvl) in sr_15m if k == "SUPPORT"]
        resists  = [lvl for (k, lvl) in sr_15m if k == "RESIST"]
        if trend_15m == "UP" and supports:
            nearest_candidate = min(supports, key=lambda s: abs(s - price))
            if abs(price - nearest_candidate) / nearest_candidate <= SR_PROX_PCT:
                side = "LONG"; nearest = nearest_candidate
        if trend_15m == "DOWN" and resists and not side:
            nearest_candidate = min(resists, key=lambda s: abs(s - price))
            if abs(price - nearest_candidate) / nearest_candidate <= SR_PROX_PCT:
                side = "SHORT"; nearest = nearest_candidate

    if not side or nearest is None:
        log(symbol, "no valid confluence/trend/NY-exception match -> skip")
        return

    # cooldown & daily cap guard
    if now_ms() - last_alert_ts[symbol] < ALERT_COOLDOWN_SEC * 1000:
        log(symbol, "cooldown active -> skip")
        return
    if daily_count[symbol] >= DAILY_MAX_SIGNALS_PER_SYMBOL:
        log(symbol, "daily cap reached -> skip")
        return

    # build and send plan
    plan = build_trade_plan(symbol, price, side, liq_total)
    reasons = [
        f"Confluence: {'yes' if confluence_levels else 'no (NY-exception)'}",
        f"15m/1h trend: {trend_15m}/{trend_1h}",
        f"Recent liq (USD): {int(liq_total):,}",
        f"Session: NY priority" if is_ny_session() else "Session: London/NY"
    ]

    text = (
        f"📢 A+ {plan['side']} Alert (Confidence: {CONFIDENCE_LABEL})\n\n"
        f"Pair: {plan['symbol']}\n"
        f"Side: {plan['side']} ✅\n"
        f"Time: {utc_now_str()}\n\n"
        f"📈 Entry: {plan['entry']}\n"
        f"❌ Stop Loss: {plan['sl']}\n"
        f"🎯 Take Profit: {plan['tp']}\n"
        f"⚖ Risk/Reward: {plan['rr']}\n\n"
        f"🔎 Reasons:\n- " + "\n- ".join(reasons) + f"\n- SR nearest: {round(nearest, 6)}\n\n"
        f"📝 Note: Signal only — backtest & paper-trade before live execution.\n"
        f"🔁 Policy: one alert per setup | daily cap per symbol: {DAILY_MAX_SIGNALS_PER_SYMBOL}\n"
    )

    await send_telegram(http_session, text)
    last_alert_ts[symbol] = now_ms()
    daily_count[symbol] += 1
    log("ALERT SENT for", symbol, "| daily_count:", daily_count[symbol])

# -------------------------
# Binance listeners (liquidations & ticker)
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
                                qty = 0.0; price = 0.0
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
                                qty = 0.0; price = 0.0
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
# Minute aggregator -> build minute closes and evaluate
# -------------------------
async def minute_aggregator(trade_queue: asyncio.Queue, http_session):
    last_minute = None
    minute_last_price = {s: None for s in SYMBOLS}
    heartbeat_counter = 0
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
                # minute tick
                for s in SYMBOLS:
                    p = minute_last_price.get(s)
                    if p is not None:
                        minute_closes[s].append(p)
                minute_last_price = {s: None for s in SYMBOLS}
                last_minute = m
                heartbeat_counter += 1
                if heartbeat_counter % 5 == 0:
                    log("HEARTBEAT - minute ticks processed:", heartbeat_counter)
                # evaluate each symbol
                for s in SYMBOLS:
                    asyncio.create_task(evaluate_symbol(s, http_session))
        except Exception as e:
            log("[MIN-AGG] error:", e)
            await asyncio.sleep(0.1)

# -------------------------
# Entrypoint
# -------------------------
async def main():
    log("Starting A+ Futures Bot (Confluence+TOD+FixedRisk) ...")
    trade_queue = asyncio.Queue()
    async with aiohttp.ClientSession() as http_sess:
        # startup Telegram ping
        try:
            await send_telegram(http_sess, "🚀 A+ Strategy Bot (Confluence+TOD) is now LIVE ✅")
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

