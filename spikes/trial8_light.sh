#!/bin/bash
# 試行8: 板の力の下限を 8 N → 2 N にした後の、軽い本の確認（表面は結果に効かなかったので rough だけ）
set -u
N=${1:-16}
CELLS="P10:i:center,P10:ii:center,P10:i:short,P07:ii:center,P13:ii:center"
for sit in stack flat; do
  for b in book-09 book-07 book-05 book-03; do
    python spikes/trial8_pinch.py --book $b --surface rough --situation $sit --cells $CELLS --n-envs $N 2>&1 | grep "run_id\|Error"
  done
done
