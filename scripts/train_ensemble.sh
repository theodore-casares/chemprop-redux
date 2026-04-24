#!/usr/bin/env bash
# Train a 10-model ensemble. Each seed → unique 80/10/10 random split.
# Leaves runs/sd1/ untouched; writes only under runs/sd1_ens/.
#
# Usage:  bash scripts/train_ensemble.sh
#         SEEDS="0 1 2" DATA=data/sd1_train.csv OUT=runs/sd1_ens bash scripts/train_ensemble.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-.venv/bin/python}"
DATA="${DATA:-data/sd1_train.csv}"
OUT="${OUT:-runs/sd1_ens}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EPOCHS="${EPOCHS:-30}"

if [[ "$OUT" == "runs/sd1" ]]; then
  echo "refusing to write ensemble into runs/sd1 (would clobber single-model baseline)" >&2
  exit 1
fi

mkdir -p "$OUT"

for s in $SEEDS; do
  sub="$OUT/seed${s}"
  if [[ -f "$sub/model.pt" ]]; then
    echo "[seed $s] checkpoint exists at $sub/model.pt — skipping"
    continue
  fi
  echo "=========================================="
  echo " seed=$s  ->  $sub"
  echo "=========================================="
  "$PY" scripts/train.py \
    --data_path "$DATA" \
    --save_dir "$sub" \
    --epochs "$EPOCHS" \
    --seed "$s"
done

echo
echo "done. checkpoints:"
ls -1 "$OUT"/seed*/model.pt
