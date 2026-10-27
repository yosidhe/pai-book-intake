#!/bin/bash
# 試行8 の本試験（試行5 と同じ組: 5 冊 × 表面 2 × 置き方 2、各 16 環境）。使い方: bash spikes/trial8_sweep.sh [n_envs]
set -u
N=${1:-16}
CELLS="P10:i:center,P10:ii:center,P10:i:short,P07:i:center,P13:i:center,P07:ii:center,P13:ii:center"
for sit in stack flat; do
  for b in book-10 book-09 book-07 book-05 book-03; do
    for s in rough smooth; do
      python spikes/trial8_pinch.py --book $b --surface $s --situation $sit --cells $CELLS --n-envs $N 2>&1 | grep "run_id\|Error"
    done
  done
done
