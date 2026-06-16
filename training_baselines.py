"""
training_baselines.py

Unified training/evaluation script for long-term time-series forecasting baselines.

Supports two modes:

1) Lightweight/local mode:
   Uses models from models_baselines.py

   Example:
   python training_baselines.py --model DLinear --data ETTh1 \
     --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
     --freq h --features M --seq_len 96 --label_len 48 --pred_len 96

2) Official mode:
   Uses official repositories through official_model_factory.py

   Example:
   python training_baselines.py --official --model iTransformer \
     --official_root ../official_models \
     --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
     --freq h --features M --seq_len 96 --label_len 48 --pred_len 96

Required local files:
   data_loader.py
   metrics.py
   models_baselines.py
   official_model_factory.py   # only needed when --official is used

Important:
   - ARIMA is handled separately and does not use epochs/patience.
   - Official T-GCN is not forced into this generic interface because accurate
     T-GCN requires a real graph adjacency matrix and its traffic-specific setup.
   - For fair comparison with your own model, explicitly set:
       --train_epochs 20 --patience 5 --batch_size 16 --learning_rate <same_lr>
"""

from __future__ import annotations

import os
import time
import random
import argparse
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_PEMS
from metrics import metric


# =============================================================================
# Basic utilities
# =============================================================================


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    v = str(v).lower()
    if v in ("yes", "true", "t", "1", "y", "on"):
        return True
    if v in ("no", "false", "f", "0", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def as_scalar(x) -> float:
    """Convert numpy scalar/array or torch scalar to safe Python float."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return float(np.nanmean(np.asarray(x, dtype=np.float64)))


# =============================================================================
# Data
# =============================================================================


def get_dataset_class(data_name: str):
    data_name_lower = str(data_name).lower()

    if data_name in ["ETTh1", "ETTh2"]:
        return Dataset_ETT_hour

    if data_name in ["ETTm1", "ETTm2"]:
        return Dataset_ETT_minute
    
    if data_name_lower in ["pems03", "pems04", "pems07", "pems08", "pems"]:
            from data_loader import Dataset_PEMS  # 确保导入了 Dataset_PEMS
            return Dataset_PEMS
    if data_name_lower in [
        "weather",
        "exchange",
        "exchange_rate",
        "electricity",
        "traffic",
        "custom",
    ]:
        return Dataset_Custom

    raise ValueError(
        "Supported data: ETTh1, ETTh2, ETTm1, ETTm2, "
        "Weather, Exchange, Electricity, Traffic, Custom"
        "PEMS03, PEMS04, PEMS07, PEMS08"
    )


def data_provider(args, flag: str):
    Data = get_dataset_class(args.data)

    timeenc = 0 if args.embed != "timeF" else 1
    is_arima = str(args.model).lower() == "arima"

    shuffle_flag = flag == "train" and not is_arima

    # Use all validation/test samples. Dropping last test batch changes metrics
    # when the number of windows is not divisible by batch_size.
    drop_last = flag == "train" and not is_arima

    dataset = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=args.pin_memory,
    )

    var_num = dataset.data_x.shape[1] if hasattr(dataset, "data_x") else "unknown"
    print(f"{flag:>5} | samples: {len(dataset)} | variables: {var_num}")
    return dataset, loader


def infer_dims(args, train_set):
    sample_x, sample_y, sample_x_mark, sample_y_mark = train_set[0]

    args.enc_in = int(sample_x.shape[-1])
    args.dec_in = int(sample_y.shape[-1])
    args.c_out = 1 if args.features == "MS" else int(sample_y.shape[-1])
    args.time_dim = int(sample_x_mark.shape[-1])

    return args


def select_target(batch_y: torch.Tensor, args) -> torch.Tensor:
    """
    Select prediction target from batch_y.

    batch_y shape:
        [B, label_len + pred_len, C]

    For M/S:
        output all c_out variables.
    For MS:
        output only the final target channel.
    """
    if args.features == "MS":
        return batch_y[:, -args.pred_len:, -args.c_out:]
    return batch_y[:, -args.pred_len:, :args.c_out]


def build_decoder_input(batch_y: torch.Tensor, args) -> torch.Tensor:
    """
    Standard encoder-decoder input:
        known label part + zero future part.

    batch_y:
        [B, label_len + pred_len, dec_in]
    return:
        [B, label_len + pred_len, dec_in]
    """
    dec_zeros = torch.zeros_like(batch_y[:, -args.pred_len:, :])
    return torch.cat([batch_y[:, :args.label_len, :], dec_zeros], dim=1)


def build_correlation_adj(train_data: np.ndarray, top_k: int = 5, use_abs: bool = True) -> torch.Tensor:
    """
    Build a correlation adjacency matrix for local lightweight T-GCN.

    This is NOT a replacement for official T-GCN's road-network adjacency.
    Use it only for local/lightweight T-GCN baselines.
    """
    if train_data.ndim != 2:
        raise ValueError(f"Expected train_data with shape [T, N], got {train_data.shape}")

    _, N = train_data.shape
    if N == 1:
        return torch.eye(1, dtype=torch.float32)

    x = np.asarray(train_data, dtype=np.float64)
    corr = np.corrcoef(x.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    if use_abs:
        corr = np.abs(corr)

    np.fill_diagonal(corr, 0.0)

    k = int(max(1, min(top_k, N - 1)))
    adj = np.zeros_like(corr, dtype=np.float32)
    top_idx = np.argsort(corr, axis=1)[:, -k:]

    rows = np.arange(N)[:, None]
    adj[rows, top_idx] = corr[rows, top_idx].astype(np.float32)

    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 1.0)

    return torch.tensor(adj, dtype=torch.float32)


# =============================================================================
# Model building
# =============================================================================


def build_model(args, device):
    if args.official:
        if args.auto_download_official:
            from official_repo_downloader import ensure_repo_for_model
            ensure_repo_for_model(
                model_name=args.model,
                official_root=args.official_root,
                depth=args.git_depth,
            )

        from official_model_factory import get_official_model
        model = get_official_model(args)
        print(f"Using official implementation for model: {args.model}")
    else:
        from models_baselines import get_model
        model = get_model(args)
        print(f"Using lightweight/local implementation for model: {args.model}")

    return model.to(device)

# =============================================================================
# Training and evaluation
# =============================================================================


def train_one_epoch(model, train_loader, optimizer, criterion, args, device) -> float:
    model.train()
    losses = []

    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = build_decoder_input(batch_y, args).float().to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        target = select_target(batch_y, args)

        loss = criterion(outputs, target)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        losses.append(loss.item())

    if len(losses) == 0:
        raise RuntimeError("No training batches were produced. Reduce batch_size or check dataset length.")

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, data_loader, criterion, args, device) -> Tuple[float, Tuple[float, ...], np.ndarray, np.ndarray]:
    model.eval()

    losses = []
    preds = []
    trues = []

    for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = build_decoder_input(batch_y, args).float().to(device)

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        target = select_target(batch_y, args)

        loss = criterion(outputs, target)
        losses.append(loss.item())

        preds.append(outputs.detach().cpu().numpy())
        trues.append(target.detach().cpu().numpy())

    if len(preds) == 0:
        raise RuntimeError("No evaluation batches were produced. Check dataset length and batch_size.")

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    metrics = metric(preds, trues)
    return float(np.mean(losses)), metrics, preds, trues


def adjust_lr(optimizer, epoch: int, args):
    if args.lradj == "none":
        return

    if args.lradj == "type1":
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))

    elif args.lradj == "type2":
        schedule = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
        if epoch not in schedule:
            return
        lr = schedule[epoch]

    elif args.lradj == "cosine":
        lr = args.learning_rate * 0.5 * (1.0 + np.cos(np.pi * epoch / args.train_epochs))

    else:
        return

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print(f"Learning rate adjusted to {lr:.8f}")


def run_neural(args, train_set, train_loader, val_loader, test_loader, device):
    model = build_model(args, device)

    # Local lightweight T-GCN can use correlation adjacency.
    # Official T-GCN is intentionally not forced here.
    if (not args.official) and hasattr(model, "set_static_adj"):
        static_adj = build_correlation_adj(
            train_set.data_x,
            top_k=args.top_k,
            use_abs=args.corr_abs,
        )
        model.set_static_adj(static_adj.to(device))
    else:
        static_adj = None

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Trainable parameters: {params / 1e6:.3f}M")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    mode_tag = "official" if args.official else "local"
    setting = (
        f"{mode_tag}_{args.model}_{args.data}"
        f"_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        f"_dm{args.d_model}_el{args.e_layers}"
        f"_split{args.train_ratio:.2f}-{args.val_ratio:.2f}-{args.test_ratio:.2f}"
        f"_seed{args.seed}"
    )

    ckpt_dir = os.path.join(args.checkpoints, setting)
    result_dir = os.path.join(args.results, setting)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    checkpoint_path = os.path.join(ckpt_dir, "checkpoint.pth")

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\nStart training")
    for epoch in range(1, args.train_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            args,
            device,
        )

        val_loss, val_metrics, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            args,
            device,
        )

        val_mae, val_mse, *_ = val_metrics
        val_mae = as_scalar(val_mae)
        val_mse = as_scalar(val_mse)

        print(
            f"Epoch {epoch:03d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"MAE {val_mae:.6f} | "
            f"MSE {val_mse:.6f} | "
            f"Time {time.time() - epoch_start:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

            print(f"  Saved best checkpoint at epoch {epoch}.")
        else:
            patience_counter += 1
            print(f"  EarlyStopping counter: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

        adjust_lr(optimizer, epoch, args)

    if not os.path.exists(checkpoint_path):
        raise RuntimeError("No checkpoint was saved. Check validation loop or dataset.")

    print("\nLoading best checkpoint and testing...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics, preds, trues = evaluate(
        model,
        test_loader,
        criterion,
        args,
        device,
    )

    save_results(
        result_dir=result_dir,
        setting=setting,
        best_epoch=best_epoch,
        test_loss=test_loss,
        test_metrics=test_metrics,
        preds=preds,
        trues=trues,
        static_adj=static_adj,
    )


# =============================================================================
# ARIMA branch
# =============================================================================


def evaluate_arima(test_loader, args):
    from models_baselines import ARIMAForecaster

    forecaster = ARIMAForecaster(
        order=tuple(args.arima_order),
        fallback=args.arima_fallback,
    )

    preds = []
    trues = []

    start = time.time()
    seen = 0

    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        batch_x_np = batch_x.numpy().astype(np.float32)

        pred = forecaster.forecast_batch(
            batch_x_np,
            args.pred_len,
            args.c_out,
        )

        true = select_target(batch_y.float(), args).numpy().astype(np.float32)

        preds.append(pred)
        trues.append(true)

        seen += batch_x_np.shape[0]

        if args.max_arima_samples > 0 and seen >= args.max_arima_samples:
            break

        if args.arima_log_every > 0 and seen % args.arima_log_every == 0:
            print(f"ARIMA processed {seen} windows, elapsed {(time.time() - start) / 60:.2f} min")

    if len(preds) == 0:
        raise RuntimeError("No ARIMA test samples were produced.")

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    return metric(preds, trues), preds, trues


# =============================================================================
# Results
# =============================================================================


def save_results(
    result_dir,
    setting,
    best_epoch,
    test_loss,
    test_metrics,
    preds,
    trues,
    static_adj=None,
):
    mae, mse, rmse, mape, mspe, rse, corr = test_metrics

    mae = as_scalar(mae)
    mse = as_scalar(mse)
    rmse = as_scalar(rmse)
    mape = as_scalar(mape)
    mspe = as_scalar(mspe)
    rse = as_scalar(rse)
    corr = as_scalar(corr)
    test_loss = as_scalar(test_loss)

    print("\nTest result")
    print(f"Loss: {test_loss:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.6f}")
    print(f"MSPE: {mspe:.6f}")
    print(f"RSE : {rse:.6f}")
    print(f"CORR: {corr:.6f}")

    os.makedirs(result_dir, exist_ok=True)

    np.save(os.path.join(result_dir, "pred.npy"), preds)
    np.save(os.path.join(result_dir, "true.npy"), trues)

    if static_adj is not None:
        np.save(
            os.path.join(result_dir, "static_adj.npy"),
            static_adj.detach().cpu().numpy(),
        )

    with open(os.path.join(result_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"setting: {setting}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"test_loss: {test_loss:.8f}\n")
        f.write(f"mae: {mae:.8f}\n")
        f.write(f"mse: {mse:.8f}\n")
        f.write(f"rmse: {rmse:.8f}\n")
        f.write(f"mape: {mape:.8f}\n")
        f.write(f"mspe: {mspe:.8f}\n")
        f.write(f"rse: {rse:.8f}\n")
        f.write(f"corr: {corr:.8f}\n")

    print(f"\nResults saved to: {result_dir}")


# =============================================================================
# Main
# =============================================================================


def main(args):
    set_seed(args.seed, deterministic=args.deterministic)

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must be 1.0, got {ratio_sum}."
        )

    if args.official and str(args.model).lower() == "arima":
        print("Warning: --official is ignored for ARIMA.")

    if args.official and str(args.model).lower() in ["tgcn", "t-gcn"]:
        print(
            "Warning: official T-GCN is not supported by the generic LTSF interface. "
            "Use official T-GCN's own graph pipeline for accurate results."
        )

    if torch.cuda.is_available() and "cuda" in args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(
        f"Split ratio | train={args.train_ratio:.2f}, "
        f"val={args.val_ratio:.2f}, test={args.test_ratio:.2f}"
    )

    train_set, train_loader = data_provider(args, "train")
    val_set, val_loader = data_provider(args, "val")
    test_set, test_loader = data_provider(args, "test")

    args = infer_dims(args, train_set)

    print(
        f"Dims | enc_in={args.enc_in}, dec_in={args.dec_in}, c_out={args.c_out}, "
        f"time_dim={args.time_dim}, seq_len={args.seq_len}, "
        f"label_len={args.label_len}, pred_len={args.pred_len}"
    )

    if str(args.model).lower() == "arima":
        setting = (
            f"ARIMA_{args.data}"
            f"_sl{args.seq_len}_pl{args.pred_len}"
            f"_order{tuple(args.arima_order)}"
            f"_seed{args.seed}"
        )

        result_dir = os.path.join(args.results, setting)
        os.makedirs(result_dir, exist_ok=True)

        test_metrics, preds, trues = evaluate_arima(test_loader, args)
        save_results(
            result_dir=result_dir,
            setting=setting,
            best_epoch=0,
            test_loss=test_metrics[1],
            test_metrics=test_metrics,
            preds=preds,
            trues=trues,
        )
    else:
        run_neural(
            args=args,
            train_set=train_set,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )


# =============================================================================
# Arguments
# =============================================================================


def get_args():
    parser = argparse.ArgumentParser(
        description="Unified LTSF baseline training script with official-model support"
    )

    # -------------------------------------------------------------------------
    # Model selection
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        default="DLinear",
        help=(
            "ARIMA, Informer, Autoformer, TGCN, TimeMachine, SMamba, "
            "iTransformer, DLinear, PatchTST, Gateformer, PeriodNet"
        ),
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="Use official repository implementation via official_model_factory.py",
    )
    parser.add_argument(
        "--official_root",
        type=str,
        default="../official_models",
        help="Path containing official repositories",
    )
    parser.add_argument("--auto_download_official", action="store_true",
                    help="automatically git clone official repo if missing")
    parser.add_argument("--git_depth", type=int, default=1,
                    help="git clone depth. Use 0 for full clone")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)

    # -------------------------------------------------------------------------
    # Forecasting lengths and dimensions
    # -------------------------------------------------------------------------
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)

    # These are overwritten by infer_dims(), but official repos often expect
    # these fields to exist before model construction.
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--time_dim", type=int, default=4)

    # -------------------------------------------------------------------------
    # Common model hyperparameters
    # -------------------------------------------------------------------------
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--factor", type=int, default=5)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", action="store_true")

    # Informer/Autoformer official-style options
    parser.add_argument("--output_attention", action="store_true")
    parser.add_argument("--distil", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--attn", type=str, default="prob")
    parser.add_argument("--mix", type=str2bool, nargs="?", const=True, default=True)

    # THUML/newer official repo options
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--model_id", type=str, default="official_baseline")
    parser.add_argument("--class_strategy", type=str, default="projection")
    parser.add_argument("--use_norm", type=int, default=1)

    # PatchTST
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--padding_patch", type=str, default="end")
    parser.add_argument("--revin", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--affine", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--subtract_last", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--decomposition", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--kernel_size", type=int, default=25)

    # Mamba-family
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--channel_independence", type=str2bool, nargs="?", const=True, default=False)

    # T-GCN / graph for local lightweight implementation
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--corr_abs", action="store_true", default=True)
    parser.add_argument("--signed_corr", dest="corr_abs", action="store_false")

    # PeriodNet
    parser.add_argument("--period_top_k", type=int, default=3)

    # -------------------------------------------------------------------------
    # ARIMA
    # -------------------------------------------------------------------------
    parser.add_argument("--arima_order", type=int, nargs=3, default=[2, 0, 2], help="p d q")
    parser.add_argument("--arima_fallback", type=str, default="last", choices=["last", "mean"])
    parser.add_argument(
        "--max_arima_samples",
        type=int,
        default=-1,
        help="Debug speed limit. -1 means all test windows.",
    )
    parser.add_argument("--arima_log_every", type=int, default=50)

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=20,
        help="Maximum epochs per independent run",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop if validation loss does not improve for this many epochs",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lradj", type=str, default="none", choices=["none", "type1", "type2", "cosine"])

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints_baselines/")
    parser.add_argument("--results", type=str, default="./results_baselines/")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)