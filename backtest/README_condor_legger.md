# condor_legger.py — trend-led "zero-risk condor" by legging in

## The idea

A long condor bought in one go always costs money. You can end up with one at
**zero (or negative) net cost** only by entering it in two steps, with the
market moving your way in between:

| Step | Bullish example (Bank Nifty ~56,000, W = 2000, S = 500) |
|---|---|
| 1. Trend signal | SELL 58000 PE, BUY 56000 PE (wide bull put spread, credit ≈ 1,330, risk ≈ 670 pts) |
| 2. Index rises ~S | BUY 58500 PE, SELL 56500 PE at roughly the same prices step 1 got |
| Result | Long put condor 56000 / 56500 / 58000 / 58500. Net cost ≤ 0 → worst case ₹0, best case S − net cost |

Bearish signals do the mirror image with calls.

**This is not free money.** Between step 1 and step 2 you carry the full risk
of the step 1 spread (W − credit, ≈ ₹20k per Bank Nifty lot). The "zero risk"
condor is the reward for the trend being right. If the trend is wrong, the
stop-loss or trend-flip exit takes the loss.

## Trend signal

Ported from the TradingView scripts in `project-alpha/tradingview`:

* Gamma's volumetric VIDYA baseline (`vidya_calc`, length 10 / momentum 20)
* Beta's MACD alignment (MACD above signal = bullish), default 12/26/9 because
  this runs on 60-minute or daily bars, not 1-minute
* Gamma's max-entry-distance filter (no entry if price is stretched far from
  the baseline)

Bullish = close > VIDYA **and** MACD > signal **and** not stretched. Bearish is
the opposite. The trade exits early if price crosses back over the baseline.

## Rules (defaults, all overridable with `--set k=v`)

| Rule | Bank Nifty | Nifty |
|---|---|---|
| Step 1 width `W` | 2000 | 800 |
| Lock shift `S` | 500 | 200 |
| Lock when net cost ≤ `lock_cost` | 0 | 0 |
| Step 1 stop-loss `sl` (pts) | 350 | 140 |
| Expiry chosen | first with 10–35 days left | same |
| Unlocked: exit when ≤ 3 days left | ✓ | ✓ |
| Locked: take profit at `target_frac` of max | 60% | 60% |
| Locked: exit `exit_dte` days before expiry | 1 | 1 |
| No re-entry in the same direction after an exit until the signal resets | ✓ | ✓ |

## Usage

```bash
# 1. Backtest (options are Black-Scholes priced from India VIX, ±slip per leg)
python3 condor_legger.py backtest BANKNIFTY banknifty_60m.json --vix vix_60m.json -v
python3 condor_legger.py backtest NIFTY nifty_day.json --vix vix_day.json --set W=600 S=200

# 2. Paper trading: live Kite quotes, prints the orders it *would* place
pip install kiteconnect
export KITE_API_KEY=... KITE_ACCESS_TOKEN=...
python3 condor_legger.py paper BANKNIFTY --tf 60minute

# 3. Live (asks you to type YES first)
python3 condor_legger.py live BANKNIFTY --lots 1
```

Candle files are Kite historical JSON (a list of `{date, open, high, low,
close}` or the MCP wrapper `{"data": [...]}`), same as `backtest.py`.

## Limits — read before using real money

* **Backtest option prices are modelled.** Kite does not serve history for
  expired option contracts, so fills come from Black-Scholes with IV = India
  VIX and a fixed slippage per leg. Real skew, IV spikes and wide bid-ask on
  deep ITM strikes will differ. Run `paper` for several expiries first and
  compare its journal (`condor_<INDEX>_<date>.csv`) with the backtest.
* **Live mode** sends LIMIT orders at the current bid/ask, buy legs first (so a
  failed leg never leaves a naked short), waits up to ~20 s per leg, and stops
  with an error if a leg does not fill. It does not chase or re-price — check
  your positions if it stops.
* Holidays are not modelled in backtest expiries; live mode uses Kite's real
  instrument list.
* Charges in the backtest are a rough ₹390 per locked trade (₹195 if exited
  before locking).
