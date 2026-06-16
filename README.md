# Unified LTSF Baselines

Files:

- `training_baselines.py`: one training/evaluation entrypoint. Select model with `--model`.
- `models_baselines.py`: all baseline classes in one file.
- `run_all_models_example.sh`: example loop over models and datasets.
- `data_loader.py`, `metrics.py`, `timefeatures.py`: copied from the user's current project for identical data processing and metrics.

Supported `--model` values:

`ARIMA`, `Informer`, `Autoformer`, `TGCN`, `TimeMachine`, `SMamba`, `iTransformer`, `DLinear`, `PatchTST`, `Gateformer`, `PeriodNet`.

Example:

```bash
python training_baselines.py \
  --model DLinear \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96
```

ARIMA is slow on multivariate datasets, especially Traffic. For debugging:

```bash
python training_baselines.py --model ARIMA --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --batch_size 1 --max_arima_samples 20
```

Notes:

- Neural models share the same PyTorch training loop.
- ARIMA is a statsmodels-based non-neural branch.
- TGCN uses correlation adjacency from the training split by default. Replace it with a real road-network adjacency matrix for Traffic if available.
- TimeMachine and S-Mamba use dependency-free Mamba-like blocks, so they do not require `mamba-ssm` CUDA extensions.
