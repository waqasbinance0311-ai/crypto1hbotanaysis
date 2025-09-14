# -------------------------
# Build trade plan (LONG/SHORT with fixed SL-TP distances)
# -------------------------
def build_trade_plan(symbol, price, side, liq_total):
    entry = float(price)
    
    if side == "LONG":
        sl = entry - 2.5
        tp = entry + 8.5
    else:  # SHORT
        sl = entry + 2.5
        tp = entry - 8.5

    def r(v):
        if v >= 1:
            return round(v, 2)
        return round(v, 4)

    return {
        "symbol": symbol,
        "side": side,
        "entry": r(entry),
        "sl": r(sl),
        "tp": r(tp),
        "rr": "1:3.4",   # fixed ratio
        "liq_usd": int(liq_total)
    }
