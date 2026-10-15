"""Python mirror of universal_stock_intraday.pine for backtesting on Kite candles.

Usage: python backtest.py SYMBOL candles.json [more.json ...] [--tf 3] [--htf 15]
Candle JSON: list of {date, open, high, low, close, volume} (Kite format), or
the MCP wrapper {"data": [...]}. Multiple files are concatenated.
"""
import json, sys, argparse
import numpy as np, pandas as pd

P = dict(emaF=20, emaS=50, adxLen=14, adxMin=20.0, atrLen=14, squeeze=0.8,
         slATR=1.5, targetR=2.0, trailATR=2.0, riskPct=0.5, maxPosX=1.0,
         win=("09:45", "14:30"), sq=(15, 10), maxTrades=3, dayLossR=2.0,
         coolBars=5, allowShort=True, capital=100000.0, costMult=1.0)

def load(files):
    rows = []
    for f in files:
        d = json.load(open(f))
        if isinstance(d, dict):
            d = d.get("data", d.get("candles", d))
        if isinstance(d, dict):
            d = d.get("candles", [])
        rows += d
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.drop_duplicates("date").sort_values("date").set_index("date")[["open","high","low","close","volume"]].astype(float)

def rma(s, n):
    return s.ewm(alpha=1/n, adjust=False).mean()

def indicators(df, htf_min, p):
    c, h, l = df.close, df.high, df.low
    df["emaF"] = c.ewm(span=p["emaF"], adjust=False).mean()
    df["emaS"] = c.ewm(span=p["emaS"], adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = rma(tr, p["atrLen"])
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    trr = rma(tr, p["adxLen"])
    pdi = 100 * rma(pd.Series(pdm, df.index), p["adxLen"]) / trr
    mdi = 100 * rma(pd.Series(mdm, df.index), p["adxLen"]) / trr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx"] = rma(dx.fillna(0), p["adxLen"])
    day = df.index.normalize()
    hlc3 = (h + l + c) / 3
    df["vwap"] = (hlc3 * df.volume).groupby(day).cumsum() / df.volume.groupby(day).cumsum().replace(0, np.nan)
    # HTF EMAs of the previous *completed* HTF bar (mirrors [1] + lookahead_on)
    htf = c.resample(f"{htf_min}min", origin="start_day", offset="15min").last().dropna()
    hF = htf.ewm(span=p["emaF"], adjust=False).mean().shift(1)
    hS = htf.ewm(span=p["emaS"], adjust=False).mean().shift(1)
    key = (df.index - pd.Timedelta(minutes=15)).floor(f"{htf_min}min") + pd.Timedelta(minutes=15)
    df["htfF"] = hF.reindex(key).values
    df["htfS"] = hS.reindex(key).values
    df["atrAvg"] = df.atr.rolling(100).mean()
    return df

def cost(px, q, is_sell):
    t = px * q
    return P["costMult"] * ( min(t * 0.0003, 20) + t * 0.00006 + (t * 0.00025 if is_sell else 0))

def run(df, tick, p):
    trades = []
    pos = 0; entry = stop = tgt = best = rps = 0.0; q = 0; risk_rs = 0.0; etime = None
    cur_day = None; n_today = 0; dayR = 0.0; last_exit = -10**9
    wlo, whi = [pd.Timestamp(f"2000-01-01 {x}").time() for x in p["win"]]
    rows = df.itertuples()
    for i, r in enumerate(rows):
        t = r.Index
        if t.date() != cur_day:
            cur_day, n_today, dayR = t.date(), 0, 0.0
        sq = (t.hour, t.minute) >= p["sq"]
        # 1) exits on this bar using stop/target set at previous close
        if pos != 0:
            px = why = None
            if pos == 1:
                if r.low <= stop: px, why = min(stop, r.open), "STOP"
                elif not np.isnan(tgt) and r.high >= tgt: px, why = max(tgt, r.open), "TARGET"
            else:
                if r.high >= stop: px, why = max(stop, r.open), "STOP"
                elif not np.isnan(tgt) and r.low <= tgt: px, why = min(tgt, r.open), "TARGET"
            if why is None and sq: px, why = r.close, "EOD"
            if why:
                px = px - pos * tick  # 1-tick slippage against us
                gross = pos * (px - entry) * q
                c = cost(entry, q, pos == -1) + cost(px, q, pos == 1)
                net = gross - c
                trades.append(dict(entry_time=etime, exit_time=t, side="L" if pos == 1 else "S", qty=q,
                                   entry=entry, exit=px, gross=gross, costs=c, net=net, R=net / risk_rs, why=why))
                dayR += net / risk_rs; last_exit = i; pos = 0
        # 2) entries at this bar's close
        regime = r.adx >= p["adxMin"] and not np.isnan(r.atrAvg) and r.atr >= p["squeeze"] * r.atrAvg
        inwin = wlo <= t.time() < whi
        can = pos == 0 and inwin and not sq and n_today < p["maxTrades"] and dayR > -p["dayLossR"] \
              and i - last_exit > p["coolBars"] and regime and not np.isnan(r.htfF)
        if can:
            up = r.emaF > r.emaS and r.close > r.vwap and r.htfF > r.htfS
            dn = r.emaF < r.emaS and r.close < r.vwap and r.htfF < r.htfS
            lt = r.low <= r.emaF < r.close and r.close > r.open
            st = r.high >= r.emaF > r.close and r.close < r.open
            dist = p["slATR"] * r.atr
            qq = int(min(p["capital"] * p["riskPct"] / 100 / dist, p["capital"] * p["maxPosX"] / r.close))
            side = 1 if (up and lt) else (-1 if (p["allowShort"] and dn and st) else 0)
            if qq > 0 and side:
                pos, q, rps = side, qq, dist
                entry = r.close + side * tick
                stop = entry - side * dist
                tgt = entry + side * p["targetR"] * dist if p["targetR"] > 0 else np.nan
                best = r.close; risk_rs = dist * qq; etime = t; n_today += 1
                continue
        # 3) trail at close (applies from next bar)
        if pos == 1:
            best = max(best, r.high)
            if best - entry >= rps: stop = max(stop, entry, best - p["trailATR"] * r.atr)
        elif pos == -1:
            best = min(best, r.low)
            if entry - best >= rps: stop = min(stop, entry, best + p["trailATR"] * r.atr)
    return pd.DataFrame(trades)

def summary(sym, tr, days, cap):
    if tr.empty:
        return dict(symbol=sym, days=days, trades=0)
    w = tr[tr.net > 0]; l = tr[tr.net <= 0]
    eq = cap + tr.net.cumsum()
    dd = (eq.cummax() - eq).max()
    daily = tr.groupby(tr.exit_time.dt.date).net.sum()
    return dict(symbol=sym, days=days, trades=len(tr), win_pct=round(100 * len(w) / len(tr), 1),
                gross=round(tr.gross.sum()), costs=round(tr.costs.sum()), net=round(tr.net.sum()),
                net_pct=round(100 * tr.net.sum() / cap, 2),
                profit_factor=round(w.net.sum() / -l.net.sum(), 2) if len(l) and l.net.sum() < 0 else float("inf"),
                avg_R=round(tr.R.mean(), 3), max_dd=round(dd), green_days=f"{(daily > 0).sum()}/{len(daily)}",
                exits=tr.why.value_counts().to_dict())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol"); ap.add_argument("files", nargs="+")
    ap.add_argument("--tf", type=int, default=3); ap.add_argument("--htf", type=int, default=15)
    ap.add_argument("--tick", type=float, default=0.1); ap.add_argument("--resample", type=int, default=0); ap.add_argument("--trades", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override params k=v")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("="); P[k] = type(P[k])(eval(v)) if not isinstance(P[k], bool) else v == "True"
    raw = load(a.files)
    if a.resample:
        raw = raw.resample(f"{a.resample}min", origin="start_day", offset="15min").agg(
            dict(open="first", high="max", low="min", close="last", volume="sum")).dropna()
    df = indicators(raw, a.htf, P)
    tr = run(df, a.tick, P)
    print(json.dumps(summary(a.symbol, tr, df.index.normalize().nunique(), P["capital"]), default=str))
    if a.trades and not tr.empty:
        print(tr.to_string())
