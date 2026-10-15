#!/bin/bash
# usage: run_all.sh <tf-suffix> [--set k=v ...]
S=$(cd "$(dirname "$0")" && pwd)
cd $S/data; sfx=$1; shift
declare -A TK=([HDFCBANK]=0.05 [INFY]=0.05 [BAJFINANCE]=0.05 [MARUTI]=1 [ITC]=0.05 [SBIN]=0.05)
for f in *_${sfx}.json; do s=${f%_*}; python3 -I $S/backtest.py $s $f --tick ${TK[$s]:-0.1} "$@"; done
