#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_all.sh /path/to/your.mat
MAT_SRC=${1:-}
if [[ -z "$MAT_SRC" ]]; then
  echo "Usage: bash run_all.sh /path/to/data.mat" >&2
  exit 1
fi
if [[ ! -f "$MAT_SRC" ]]; then
  echo "File not found: $MAT_SRC" >&2
  exit 1
fi

# The original scripts hard-code this default path.
MAT_DST="/mnt/data/3f82e08c-9373-4134-b811-015b645b3a9e.mat"
ln -sf "$MAT_SRC" "$MAT_DST"

echo "[INFO] Linked $MAT_SRC -> $MAT_DST"

cd "$(dirname "$0")/code"

python -u run_baseline_min.py | tee "../results/baseline_out.txt"
python -u run_dtw_clsmin_sweep.py | tee "../results/clsmin_sweep_out.txt"
python -u run_dtw_multi_sweep.py | tee "../results/multi_sweep_out.txt"

# Optional: DBA-enhanced ClsMin experiment (does NOT modify your original scripts)
python -u run_dtw_clsmin_dba.py | tee "../results/clsmin_dba_out.txt"

echo "[DONE] Outputs saved under dtw_cnn_handoff/results/"
