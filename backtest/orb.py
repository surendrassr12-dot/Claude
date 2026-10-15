"""Opening Range Breakout (Zarattini/Barbon/Aziz 2024 rules) on NSE stocks.

Rules per stock per day:
  * Opening range (OR) = first `or_min` minutes (09:15 -> 09:15+or_min).
  * OR candle green -> buy-stop at OR high; red -> sell-stop at OR low; doji -> no trade.
  * Stop = stop_frac x 14-day daily ATR from entry. No target. Exit at square-off time.
  * Size: risk risk_pct of capital, max lev x capital.
  * "In play" filter: OR volume / avg OR volume of prior 14 days >= rv_min.
  * Costs: Zerodha MIS (brokerage 0.03% cap Rs20/order, STT 0.025% sell, ~0.006%/side other)
    + 1 tick slippage on entry and exit.
Execution is conservative: if the entry bar also touches the stop, it counts as stopped.
"""
import json, sys, argparse
import numpy as np, pandas as pd

def load(f):
    d = json.load(open(f)); d = d.get("data", d) if isinstance(d, dict) else d
    df = pd.DataFrame(d); df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float).sort_index()

def cost(px, q, is_sell, mult=1.0):
    t = px * q
    return mult * (min(t * 0.0003, 20) + t * 0.00006 + (t * 0.00025 if is_sell else 0))

def daily_atr(daily_src, n=14):
    d = daily_src.resample("1D").agg(dict(high="max", low="min", close="last")).dropna()
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().shift(1)          # ATR known before today's open

def run(sym, bars, atr, tick, or_min=5, stop_frac=0.10, rv_min=0.0, risk_pct=1.0, lev=4.0,
        cap=100000.0, sq="15:15", cmult=1.0):
    trades = []
    or_end = (pd.Timestamp("2000-01-01 09:15") + pd.Timedelta(minutes=or_min)).time()
    sqt = pd.Timestamp(f"2000-01-01 {sq}").time()
    days = sorted(set(bars.index.date))
    or_vol_hist = []
    for d in days:
        b = bars[bars.index.date == d]
        orb = b[b.index.time < or_end]; rest = b[(b.index.time >= or_end)]
        if orb.empty or rest.empty:
            continue
        o, h, l, c, v = orb.open.iloc[0], orb.high.max(), orb.low.min(), orb.close.iloc[-1], orb.volume.sum()
        rv = v / np.mean(or_vol_hist[-14:]) if len(or_vol_hist) >= 14 else np.nan
        or_vol_hist.append(v)
        a = atr.get(pd.Timestamp(d), np.nan)
        if np.isnan(a) or np.isnan(rv) or rv < rv_min or c == o:
            continue
        side = 1 if c > o else -1
        trig = h if side == 1 else l
        dist = stop_frac * a
        q = int(min(cap * risk_pct / 100 / dist, cap * lev / trig))
        if q <= 0:
            continue
        pos = False
        for t, r in rest.iterrows():
            if t.time() >= sqt:
                if pos:
                    px = r.open - side * tick; why = "EOD"
                break
            if not pos:
                hit = r.high >= trig if side == 1 else r.low <= trig
                if not hit:
                    continue
                entry = (max(trig, r.open) if side == 1 else min(trig, r.open)) + side * tick
                stop = entry - side * dist; pos = True; et = t
                if (r.low <= stop) if side == 1 else (r.high >= stop):
                    px = stop - side * tick; why = "STOP"; break
                continue
            if (r.low <= stop) if side == 1 else (r.high >= stop):
                px = (min(stop, r.open) if side == 1 else max(stop, r.open)) - side * tick; why = "STOP"; break
        else:
            if pos:
                px = rest.close.iloc[-1] - side * tick; why = "EOD"
        if not pos:
            continue
        gross = side * (px - entry) * q
        cst = cost(entry, q, side == -1, cmult) + cost(px, q, side == 1, cmult)
        trades.append(dict(sym=sym, day=d, side="L" if side == 1 else "S", rv=round(rv, 2), qty=q,
                           entry=round(entry, 2), exit=round(px, 2), gross=gross, costs=cst,
                           net=gross - cst, R=(gross - cst) / (dist * q), why=why))
    return pd.DataFrame(trades)

def summ(name, tr, cap=100000.0):
    if tr.empty:
        return f"{name:44} no trades"
    eq = tr.sort_values("day").net.cumsum(); dd = (eq.cummax() - eq).max()
    w = tr[tr.net > 0]; l = tr[tr.net <= 0]
    pf = w.net.sum() / -l.net.sum() if l.net.sum() < 0 else float("inf")
    return (f"{name:44} trades={len(tr):3} win%={100*len(w)/len(tr):5.1f} gross={tr.gross.sum():8.0f} "
            f"costs={tr.costs.sum():7.0f} net={tr.net.sum():8.0f} ({100*tr.net.sum()/cap:+.1f}%) "
            f"PF={pf:4.2f} avgR={tr.R.mean():+.2f} bestR={tr.R.max():+.1f} maxDD={dd:6.0f}")
