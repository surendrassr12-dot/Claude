"""Trend-led "zero-risk condor" legger for Nifty / Bank Nifty options.

Idea (see README_condor_legger.md):
  Step 1  On a trend signal, open a wide credit spread in the trend's favour.
          Bullish: SELL put Kh, BUY put Kl (Kl = ATM, Kh = Kl + W)
          Bearish: SELL call Kl, BUY call Kh (Kh = ATM, Kl = Kh - W)
  Step 2  If the index then moves S points the right way, the same spread
          shifted by S costs about what step 1 collected. Add it in reverse:
          Bullish: BUY put Kh+S, SELL put Kl+S
          Bearish: BUY call Kl-S, SELL call Kh-S
          Net cost <= max_lock_cost (0 = zero risk) -> a long condor whose
          worst case is -(net cost) and best case is S - net cost.
  Exits   Step 1 not locked: stop-loss, trend flip, or too close to expiry.
          Locked: profit target or the day before expiry.

Trend = Gamma's volumetric VIDYA baseline (vidya_calc port) + Beta's MACD
alignment + Gamma's max-entry-distance filter, on whatever timeframe you feed.

Modes
  backtest  python condor_legger.py backtest INDEX candles.json [--vix vix.json]
            Options are MODEL-PRICED (Black-Scholes, IV from India VIX or
            --iv) because Kite has no history for expired contracts. Treat the
            result as a check of the rules, not of real fills.
  paper     python condor_legger.py paper INDEX      (live quotes, no orders)
  live      python condor_legger.py live INDEX       (places orders; asks YES)
            paper/live need `pip install kiteconnect` and env KITE_API_KEY,
            KITE_ACCESS_TOKEN.
"""
import argparse, csv, json, math, os, sys, time
from datetime import date, datetime, timedelta
import numpy as np, pandas as pd

INDEX = {  # lot, strike step, W (step-1 width), S (lock shift), slip/leg, stop pts
    "BANKNIFTY": dict(lot=30, step=100, W=2000, S=500, slip=3.0, sl=350, max_dist=900,
                      spot="NSE:NIFTY BANK", exch="NFO", weekly=False),
    "NIFTY":     dict(lot=65, step=50, W=800, S=200, slip=1.0, sl=140, max_dist=350,
                      spot="NSE:NIFTY 50", exch="NFO", weekly=True),
}
P = dict(vidya_len=10, vidya_mom=20, macd=(12, 26, 9), atr_len=14,
         min_dte=10, max_dte=35, lock_cost=0.0, exit_dte=1, target_frac=0.6,
         rate=0.06, iv=None)

# ---------- option maths ----------
def _N(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs(S, K, T, v, cp, r=0.0):
    if T <= 0 or v <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + v * v / 2) * T) / (v * math.sqrt(T)); d2 = d1 - v * math.sqrt(T)
    if cp == "C": return S * _N(d1) - K * math.exp(-r * T) * _N(d2)
    return K * math.exp(-r * T) * _N(-d2) - S * _N(-d1)

def greeks(S, K, T, v, cp):
    """delta, gamma (per point), theta (per day) — for display / sanity checks."""
    if T <= 0: return (1.0 if (cp == "C" and S > K) else -1.0 if (cp == "P" and S < K) else 0.0), 0.0, 0.0
    d1 = (math.log(S / K) + v * v / 2 * T) / (v * math.sqrt(T))
    pdf = math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
    delta = _N(d1) if cp == "C" else _N(d1) - 1
    return delta, pdf / (S * v * math.sqrt(T)), -S * pdf * v / (2 * math.sqrt(T)) / 365

# ---------- trend (ports of Gamma's vidya_calc and Beta's MACD) ----------
def vidya(src, length, mom):
    ch = src.diff()
    pos = ch.clip(lower=0).rolling(mom).sum(); neg = (-ch.clip(upper=0)).rolling(mom).sum()
    cmo = (100 * (pos - neg) / (pos + neg).replace(0, np.nan)).abs().fillna(0) / 100
    a = 2.0 / (length + 1); out = np.zeros(len(src)); prev = 0.0
    for i, (s, k) in enumerate(zip(src.values, cmo.values)):
        prev = a * k * s + (1 - a * k) * prev; out[i] = prev
    return pd.Series(out, src.index).rolling(15).mean()

def trend(df, cfg):
    c, h, l = df.close, df.high, df.low
    f, s, g = P["macd"]
    macd = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    sig = macd.ewm(span=g, adjust=False).mean()
    base = vidya(c, P["vidya_len"], P["vidya_mom"])
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / P["atr_len"], adjust=False).mean()
    near = (c - base).abs() < cfg["max_dist"]
    df["trend"] = np.where((c > base) & (macd > sig) & near, 1, np.where((c < base) & (macd < sig) & near, -1, 0))
    df["flip"] = np.where(c > base, 1, -1)  # raw side of baseline, used for exits
    return df

# ---------- expiries ----------
def last_wd(y, m, wd):
    d = date(y, m + 1, 1) - timedelta(days=1) if m < 12 else date(y, 12, 31)
    return d - timedelta(days=(d.weekday() - wd) % 7)

def expiries(start, end, weekly):
    """Tuesday expiries from Sep-2025 (Thursday before). Holidays not modelled."""
    out, d = [], start
    while d <= end + timedelta(days=60):
        wd = 1 if d >= date(2025, 9, 1) else 3
        if weekly:
            out.append(d + timedelta(days=(wd - d.weekday()) % 7)); d += timedelta(days=7)
        else:
            e = last_wd(d.year, d.month, wd)
            if e >= d: out.append(e)
            d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return sorted(set(out))

def pick_expiry(day, exps):
    for e in exps:
        if P["min_dte"] <= (e - day).days <= P["max_dte"]: return e

# ---------- the strategy state machine (shared by backtest and live) ----------
class Legger:
    def __init__(self, cfg, quote, place, log):
        self.cfg, self.quote, self.place, self.log = cfg, quote, place, log
        self.pos = None  # dict(side, exp, legs=[(qty, K, cp, price)], credit, locked)
        self.block = 0   # side just exited; no re-entry until the trend signal resets

    def legs_step1(self, side, atm):
        W = self.cfg["W"]
        if side == 1: return [(-1, atm + W, "P"), (+1, atm, "P")]
        return [(-1, atm - W, "C"), (+1, atm, "C")]

    def legs_lock(self):
        side, S, (a, b) = self.pos["side"], self.cfg["S"], self.pos["legs"][:2]
        if side == 1: return [(+1, a[1] + S, "P"), (-1, b[1] + S, "P")]
        return [(+1, a[1] - S, "C"), (-1, b[1] - S, "C")]

    def fill(self, legs, exp):  # buys at ask, sells at bid; returns legs with prices
        return [(q, K, cp, self.quote(K, cp, exp, "ask" if q > 0 else "bid")) for q, K, cp in legs]

    def mtm(self, exp):  # value if closed now (sell longs at bid, buy shorts at ask)
        return sum(q * self.quote(K, cp, exp, "bid" if q > 0 else "ask") for q, K, cp, _ in self.pos["legs"])

    def pnl(self, exp):
        return self.mtm(exp) - sum(q * px for q, _, _, px in self.pos["legs"])

    def close(self, now, why):
        exp = self.pos["exp"]; pnl = self.pnl(exp)
        self.place([(-q, K, cp) for q, K, cp, _ in self.pos["legs"]], exp)
        self.log(now, "EXIT", why, pnl, self.pos); self.block = self.pos["side"]; self.pos = None
        return pnl

    def step(self, now, spot, sig, flip, exps):
        cfg = self.cfg
        if self.pos is None:
            if sig != self.block: self.block = 0
            if sig == 0 or sig == self.block: return
            exp = pick_expiry(now.date(), exps)
            if exp is None: return
            atm = round(spot / cfg["step"]) * cfg["step"]
            legs = self.fill(self.legs_step1(sig, atm), exp)
            credit = -sum(q * px for q, _, _, px in legs)
            if credit <= 0 or cfg["W"] - credit <= 0: return  # mispriced / stale quotes
            self.place([(q, K, cp) for q, K, cp, _ in legs], exp)
            self.pos = dict(side=sig, exp=exp, legs=legs, credit=credit, locked=False, opened=now)
            self.log(now, "STEP1", f"credit {credit:.1f} risk {cfg['W'] - credit:.1f}", 0.0, self.pos)
            return
        exp, dte = self.pos["exp"], (self.pos["exp"] - now.date()).days
        if not self.pos["locked"]:
            lock = self.fill(self.legs_lock(), exp)
            debit = sum(q * px for q, _, _, px in lock)
            if debit - self.pos["credit"] <= P["lock_cost"]:
                self.place([(q, K, cp) for q, K, cp, _ in lock], exp)
                self.pos["legs"] += lock; self.pos["locked"] = True
                net = debit - self.pos["credit"]
                self.pos["max_profit"] = cfg["S"] - net
                self.log(now, "LOCK", f"net cost {net:.1f} -> worst {-max(net, 0):.1f} best {cfg['S'] - net:.1f}", 0.0, self.pos)
                return
            if self.pnl(exp) <= -cfg["sl"]: return self.close(now, "STOP_LOSS")
            if flip != self.pos["side"]: return self.close(now, "TREND_FLIP")
            if dte <= P["exit_dte"] + 2: return self.close(now, "NO_LOCK_NEAR_EXPIRY")
        else:
            if self.pnl(exp) >= P["target_frac"] * self.pos["max_profit"]: return self.close(now, "TARGET")
            if dte <= P["exit_dte"]: return self.close(now, "EXPIRY_EXIT")

# ---------- backtest ----------
def load(f):
    d = json.load(open(f))
    d = d.get("data", d) if isinstance(d, dict) else d
    df = pd.DataFrame(d); df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.drop_duplicates("date").sort_values("date").set_index("date")[["open", "high", "low", "close"]].astype(float)

def backtest(a):
    cfg = INDEX[a.index]; df = trend(load(a.candles), cfg)
    vix = load(a.vix).close.reindex(df.index, method="ffill") if a.vix else None
    exps = expiries(df.index[0].date(), df.index[-1].date(), cfg["weekly"])
    st = dict(spot=0.0, iv=0.13, now=None); trades = []

    def quote(K, cp, exp, side):
        T = max((datetime.combine(exp, datetime.min.time()) + timedelta(hours=15.5) - st["now"]).total_seconds(), 0) / (365 * 86400)
        px = bs(st["spot"], K, T, st["iv"], cp)
        return max(px + (cfg["slip"] if side == "ask" else -cfg["slip"]), 0.05)

    def log(now, ev, msg, pnl, pos):
        if ev == "EXIT": trades.append(dict(opened=pos["opened"], closed=now, side=pos["side"], locked=pos["locked"],
                                            why=msg, pnl_pts=round(pnl, 1), pnl_rs=round(pnl * cfg["lot"])))
        if a.verbose: print(f"{now:%Y-%m-%d %H:%M} {ev:6} {msg} {'' if ev != 'EXIT' else f'P&L {pnl:+.1f} pts'}")

    lg = Legger(cfg, quote, lambda legs, exp: None, log)
    for now, r in df.iloc[60:].iterrows():
        st.update(spot=r.close, now=now.replace(hour=15, minute=25) if now.hour == 0 else now,
                  iv=P["iv"] or (vix.loc[now] / 100 if vix is not None else 0.13))
        lg.step(st["now"], r.close, int(r.trend), int(r.flip), exps)
    if lg.pos: lg.close(st["now"], "END_OF_DATA")
    t = pd.DataFrame(trades)
    if t.empty: print("no trades"); return
    cost = 8 * 20 * 1.18 + 200  # brokerage+GST for up to 8 orders + rough STT/exchange
    t["net_rs"] = t.pnl_rs - np.where(t.locked, cost, cost / 2)
    print(t.to_string(index=False))
    print(f"\n{a.index}: {len(t)} trades | locked {t.locked.mean():.0%} | win {(t.net_rs > 0).mean():.0%} | "
          f"net ₹{t.net_rs.sum():,.0f} | avg ₹{t.net_rs.mean():,.0f} | worst ₹{t.net_rs.min():,.0f}")
    print("Options are model-priced; validate with paper mode before trading real money.")

# ---------- paper / live (Kite Connect) ----------
def run_live(a):
    from kiteconnect import KiteConnect
    cfg = INDEX[a.index]; kite = KiteConnect(api_key=os.environ["KITE_API_KEY"])
    kite.set_access_token(os.environ["KITE_ACCESS_TOKEN"])
    live = a.mode == "live"
    if live and input(f"LIVE orders on {a.index}, {a.lots} lot(s). Type YES: ") != "YES": sys.exit("aborted")
    ins = [i for i in kite.instruments(cfg["exch"]) if i["name"] == a.index and i["segment"].endswith("OPT")]
    sym = {(i["expiry"], float(i["strike"]), i["instrument_type"][0]): i["tradingsymbol"] for i in ins}
    exps = sorted({i["expiry"] for i in ins})
    tok = kite.ltp([cfg["spot"]])[cfg["spot"]]["instrument_token"]
    journal = open(f"condor_{a.index}_{date.today():%Y%m%d}.csv", "a", newline=""); jw = csv.writer(journal)

    def ts(K, cp, exp): return f"{cfg['exch']}:{sym[(exp, float(K), cp)]}"

    def quote(K, cp, exp, side):
        d = kite.quote([ts(K, cp, exp)])[ts(K, cp, exp)]["depth"]["buy" if side == "bid" else "sell"][0]["price"]
        if not d: raise RuntimeError(f"no {side} for {K}{cp}")
        return d

    def place(legs, exp):
        for q, K, cp in sorted(legs, key=lambda x: -x[0]):  # buys first: margin benefit, never naked
            px = quote(K, cp, exp, "ask" if q > 0 else "bid")
            print(f"  {'BUY ' if q > 0 else 'SELL'} {sym[(exp, float(K), cp)]} x{a.lots} @ {px}")
            if live:
                oid = kite.place_order(variety="regular", exchange=cfg["exch"], tradingsymbol=sym[(exp, float(K), cp)],
                                       transaction_type="BUY" if q > 0 else "SELL", quantity=abs(q) * a.lots * cfg["lot"],
                                       product="NRML", order_type="LIMIT", price=px)
                for _ in range(20):  # wait up to ~20 s for the fill before sending the next leg
                    if kite.order_history(oid)[-1]["status"] == "COMPLETE": break
                    time.sleep(1)
                else:
                    kite.cancel_order(variety="regular", order_id=oid)
                    raise RuntimeError(f"leg not filled, cancelled {oid}; check positions manually")

    def log(now, ev, msg, pnl, pos):
        print(f"{now:%H:%M:%S} {ev} {msg} P&L {pnl * cfg['lot'] * a.lots:+,.0f}")
        jw.writerow([now, ev, msg, round(pnl, 1)]); journal.flush()

    lg = Legger(cfg, quote, place, log)
    while True:
        now = datetime.now()
        if now.weekday() < 5 and (9, 20) <= (now.hour, now.minute) <= (15, 20):
            c = pd.DataFrame(kite.historical_data(tok, now - timedelta(days=200), now, a.tf))
            c["date"] = pd.to_datetime(c["date"]).dt.tz_localize(None)
            r = trend(c.set_index("date")[["open", "high", "low", "close"]], cfg).iloc[-2]  # last closed bar
            spot = kite.ltp([cfg["spot"]])[cfg["spot"]]["last_price"]
            try: lg.step(now, spot, int(r.trend), int(r.flip), exps)
            except Exception as e: print("skip:", e)
        time.sleep(a.poll)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["backtest", "paper", "live"]); ap.add_argument("index", choices=list(INDEX))
    ap.add_argument("candles", nargs="?"); ap.add_argument("--vix")
    ap.add_argument("--tf", default="60minute"); ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--lots", type=int, default=1); ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override, e.g. W=1500 S=500 sl=300 lock_cost=20")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("="); v = float(v)
        (INDEX[a.index] if k in INDEX[a.index] else P)[k] = v
    backtest(a) if a.mode == "backtest" else run_live(a)
