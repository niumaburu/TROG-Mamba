#!/usr/bin/env bash
set -e

# Put this script in the same folder as training_baselines.py.
# Adjust DATA_ROOT to your real dataset root.
DATA_ROOT=${1:-./dataset}
PRED_LEN=${2:-96}

MODELS=(ARIMA Informer Autoformer TGCN TimeMachine SMamba iTransformer DLinear PatchTST Gateformer PeriodNet)

run_one () {
  local model=$1
  local data=$2
  local root=$3
  local file=$4
  local freq=$5

  local bs=16
  if [ "$model" = "ARIMA" ]; then
    bs=1
  fi

  python training_baselines.py \
    --model "$model" \
    --data "$data" \
    --root_path "$root" \
    --data_path "$file" \
    --features M \
    --target OT \
    --freq "$freq" \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "$PRED_LEN" \
    --batch_size "$bs" \
    --train_epochs 20 \
    --patience 5 \
    --d_model 64 \
    --d_ff 256 \
    --n_heads 4 \
    --e_layers 2
}

for model in "${MODELS[@]}"; do
  run_one "$model" ETTh1 "$DATA_ROOT/ETT-small/" ETTh1.csv h
  run_one "$model" ETTh2 "$DATA_ROOT/ETT-small/" ETTh2.csv h
  run_one "$model" ETTm1 "$DATA_ROOT/ETT-small/" ETTm1.csv t
  run_one "$model" ETTm2 "$DATA_ROOT/ETT-small/" ETTm2.csv t
  run_one "$model" Traffic "$DATA_ROOT/traffic/" traffic.csv h
done
