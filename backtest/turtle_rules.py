"""Original Turtle Trading rules (Faith, 'Way of the Turtle' / the published Turtle rules), daily bars.

N        = 20-day Wilder ATR.
Unit     = 1% of equity / N  shares.
System 1 = enter on 20-day breakout; skip it if the previous S1 breakout (taken or not) was a
           winner, but then take the 55-day 'failsafe' breakout. Exit on 10-day opposite breakout.
System 2 = enter on 55-day breakout, no filter. Exit on 20-day opposite breakout.
Pyramid  = add 1 unit every +0.5 N from last entry, max 4 units.
Stop     = 2 N from the most recent entry, applied to all units.
Fills    = breakout/stop levels intraday (stop orders); gap-through fills at the open.
           If entry and stop are both touched on the same day -> assume stopped (conservative).
Modes    = 'cash'    : long-only delivery, exposure capped at 1x equity, delivery costs.
           'futures' : long + short, exposure capped at `lev` x equity, futures costs.
"""
import json, argparse
import numpy as np, pandas as pd

COST = dict(cash=0.0012, futures=0.0002)   # per side: delivery ~0.1% STT+stamp+charges; futures ~0.02%
SLIP = 0.0005                               # 0.05% per fill

def load(f):
    df = pd.DataFrame(json.load(open(f)))
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date")[["open", "high", "low", "close"]].astype(float)

def prep(df):
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(), (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    df["N"] = tr.ewm(alpha=1/20, adjust=False).mean().shift(1)       # known before today
    for n in (10, 20, 55):
        df[f"hh{n}"] = df.high.rolling(n).max().shift(1)
        df[f"ll{n}"] = df.low.rolling(n).min().shift(1)
    return df.dropna()

def backtest(df, system=2, mode="cash", lev=2.0, cap0=1_000_000.0):
    ent, ext = (20, 10) if system == 1 else (55, 20)
    cost = COST[mode]
    cash = cap0; units = []          # list of (qty, price)
    side = 0; stop = np.nan; last_add = np.nan; Nentry = np.nan
    trades = []; equity = []
    shadow = None; last_s1_winner = False   # S1 winner filter
    for t, r in df.iterrows():
        pos_qty = sum(q for q, _ in units)
        mtm = cash + side * sum(q * (r.open - p) for q, p in units) if side else cash
        # ---- S1 shadow trade for the filter (tracks every 20-day breakout, taken or not)
        if system == 1:
            if shadow is None:
                if r.high > r.hh20: shadow = (1, r.hh20, r.hh20 - 2 * r.N)
                elif r.low < r.ll20: shadow = (-1, r.ll20, r.ll20 + 2 * r.N)
            else:
                s, p, st = shadow
                if s == 1 and (r.low <= st or r.low < r.ll10):
                    last_s1_winner = max(st, r.ll10) > p; shadow = None
                elif s == -1 and (r.high >= st or r.high > r.hh10):
                    last_s1_winner = min(st, r.hh10) < p; shadow = None
        # ---- exits for real position
        if side:
            exit_lvl = r[f"ll{ext}"] if side == 1 else r[f"hh{ext}"]
            hit_stop = r.low <= stop if side == 1 else r.high >= stop
            hit_exit = r.low < exit_lvl if side == 1 else r.high > exit_lvl
            if hit_stop or hit_exit:
                lvl = (max(stop, exit_lvl) if side == 1 else min(stop, exit_lvl)) if (hit_stop and hit_exit) else (stop if hit_stop else exit_lvl)
                px = (min(lvl, r.open) if side == 1 else max(lvl, r.open)) * (1 - side * SLIP)
                gross = side * sum(q * (px - p) for q, p in units)
                c = sum(q * px for q, _ in units) * cost
                cash += gross - c
                trades.append(dict(entry=units[0][1], exit=px, side=side, units=len(units), net=gross - c - entry_cost,
                                   start=t_start, end=t, why="STOP" if hit_stop and (not hit_exit or lvl == stop) else "EXIT"))
                units = []; side = 0
        # ---- pyramiding
        if side and len(units) < 4:
            nxt = last_add + side * 0.5 * Nentry
            if (r.high >= nxt) if side == 1 else (r.low <= nxt):
                q = int(cash * 0.01 / Nentry)
                expo = (sum(qq for qq, _ in units) + q) * nxt
                if q > 0 and expo <= (1.0 if mode == "cash" else lev) * cash:
                    px = (max(nxt, r.open) if side == 1 else min(nxt, r.open)) * (1 + side * SLIP)
                    units.append((q, px)); last_add = px
                    stop = px - side * 2 * Nentry
                    c = q * px * cost; cash -= c; entry_cost += c
        # ---- entries
        if not side:
            d = 0
            if r.high > r[f"hh{ent}"]: d, lvl = 1, r[f"hh{ent}"]
            elif mode == "futures" and r.low < r[f"ll{ent}"]: d, lvl = -1, r[f"ll{ent}"]
            if system == 1 and d and last_s1_winner:
                # filtered: fall back to the 55-day failsafe breakout
                if d == 1 and r.high > r.hh55: lvl = r.hh55
                elif d == -1 and r.low < r.ll55: lvl = r.ll55
                else: d = 0
            if d:
                q = int(min(cash * 0.01 / r.N, (1.0 if mode == "cash" else lev) * cash / lvl))
                if q > 0:
                    px = (max(lvl, r.open) if d == 1 else min(lvl, r.open)) * (1 + d * SLIP)
                    side, Nentry, last_add, t_start = d, r.N, px, t
                    units = [(q, px)]; stop = px - d * 2 * r.N
                    entry_cost = q * px * cost; cash -= entry_cost
                    if (r.low <= stop) if d == 1 else (r.high >= stop):   # same-day stop-out
                        sx = stop * (1 - d * SLIP); g = d * q * (sx - px); c2 = q * sx * cost
                        cash += g - c2
                        trades.append(dict(entry=px, exit=sx, side=d, units=1, net=g - c2 - entry_cost, start=t, end=t, why="STOP"))
                        units = []; side = 0
        equity.append((t, cash + (side * sum(q * (r.close - p) for q, p in units) if side else 0)))
    eq = pd.Series(dict(equity))
    return pd.DataFrame(trades), eq

def stats(name, tr, eq, df, cap0=1_000_000.0):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / cap0) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    bh = (df.close.iloc[-1] / df.close.iloc[0]) ** (1 / yrs) - 1
    bhdd = (df.close / df.close.cummax() - 1).min()
    w = tr[tr.net > 0] if len(tr) else tr
    pf = w.net.sum() / -tr[tr.net <= 0].net.sum() if len(tr) and tr[tr.net <= 0].net.sum() < 0 else float("nan")
    return dict(name=name, final=round(eq.iloc[-1]), CAGR=f"{cagr*100:+.1f}%", maxDD=f"{dd*100:.1f}%",
                trades=len(tr), win=f"{100*len(w)/max(len(tr),1):.0f}%", PF=round(pf, 2),
                best=round(tr.net.max()) if len(tr) else 0, BH_CAGR=f"{bh*100:+.1f}%", BH_maxDD=f"{bhdd*100:.1f}%")
