"""Training script for ablation-ready SparseGraph-BiMamba forecasting.

Supports:
    PEMS03 / PEMS04 / PEMS07 / PEMS08 as NPZ datasets
    Traffic / Electricity / Solar-Energy as CSV datasets

Main ablation switches:
    --use_bimamba false
    --use_graph_diffusion false
    --use_sna false
    --use_tanh_gate false
    --use_orth_res false
"""

import os
import time
import random
import argparse
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_PEMS
from models import GCN_mamba_TSForecast
from metrics import metric


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).lower()
    if s in ["yes", "true", "t", "1", "y"]:
        return True
    if s in ["no", "false", "f", "0", "n"]:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataset_class(data_name: str):
    name = data_name.lower()
    if data_name in ["ETTh1", "ETTh2"]:
        return Dataset_ETT_hour
    if data_name in ["ETTm1", "ETTm2"]:
        return Dataset_ETT_minute
    if name in ["pems", "pems03", "pems04", "pems07", "pems08"]:
        return Dataset_PEMS
    if name in ["weather", "exchange", "exchange_rate", "electricity", "traffic", "solar", "solar-energy", "solar_energy", "custom"]:
        return Dataset_Custom
    raise ValueError("Unsupported data name.")


def data_provider(args, flag: str):
    Data = get_dataset_class(args.data)
    timeenc = 0 if args.embed != "timeF" else 1
    shuffle_flag = flag == "train"
    batch_size = args.batch_size if flag == "train" else (args.eval_batch_size or args.batch_size)
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
        value_channel=args.value_channel,
        auto_fix_split=args.auto_fix_split,
    )
    num_samples = len(dataset)
    if num_samples <= 0:
        raise ValueError(
            f"{flag} dataset has no valid windows. data_len={len(dataset.data_x)}, "
            f"seq_len={args.seq_len}, pred_len={args.pred_len}."
        )
    drop_last = bool(flag == "train" and args.drop_last and num_samples >= batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    print(f"{flag:>5} | samples: {num_samples} | variables: {dataset.data_x.shape[1]} | batch={batch_size} | drop_last={drop_last}")
    return dataset, loader


def load_adj_matrix(path: str) -> torch.Tensor:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        obj = np.load(path, allow_pickle=True)
        if "adj" in obj.files:
            arr = obj["adj"]
        elif "adj_mx" in obj.files:
            arr = obj["adj_mx"]
        else:
            arr = obj[obj.files[0]]
    elif ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, (list, tuple)):
            arr = obj[-1]
        elif isinstance(obj, dict):
            arr = obj.get("adj", obj.get("adj_mx", None))
            if arr is None:
                arr = next(iter(obj.values()))
        else:
            arr = obj
    else:
        arr = pd.read_csv(path, header=None).values
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got {arr.shape}")
    return torch.tensor(arr, dtype=torch.float32)


def build_correlation_adj(train_data: np.ndarray, top_k: int = 5, use_abs: bool = True) -> torch.Tensor:
    if train_data.ndim != 2:
        raise ValueError(f"Expected train_data [T,N], got {train_data.shape}")
    _, N = train_data.shape
    if N == 1:
        return torch.eye(1, dtype=torch.float32)
    x = np.asarray(train_data, dtype=np.float64)
    corr = np.corrcoef(x.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if use_abs:
        corr = np.abs(corr)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 0.0)
    k = int(max(1, min(top_k, N - 1)))
    adj = np.zeros_like(corr, dtype=np.float32)
    top_idx = np.argsort(corr, axis=1)[:, -k:]
    rows = np.arange(N)[:, None]
    adj[rows, top_idx] = corr[rows, top_idx].astype(np.float32)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 1.0)
    return torch.tensor(adj, dtype=torch.float32)


def maybe_find_adj_in_root(root_path: str) -> Optional[str]:
    candidates = ["adj.npy", "adj_mx.npy", "adj.npz", "adj_mx.npz", "adj.pkl", "adj_mx.pkl", "distance.csv", "distances.csv", "adj.csv"]
    for name in candidates:
        path = os.path.join(root_path, name)
        if os.path.exists(path):
            return path
    return None


def build_static_adj_for_training(args, train_set) -> torch.Tensor:
    N = int(train_set.data_x.shape[1])
    identity = torch.eye(N, dtype=torch.float32)
    adj_type = str(args.adj_type).lower()
    static_adj = None
    if args.adj_path:
        static_adj = load_adj_matrix(args.adj_path)
    elif adj_type in ["static", "hybrid"]:
        found = maybe_find_adj_in_root(args.root_path)
        if found is not None:
            print(f"Found adjacency file: {found}")
            static_adj = load_adj_matrix(found)
    corr_adj = None
    if adj_type in ["correlation", "corr", "hybrid"]:
        corr_adj = build_correlation_adj(train_set.data_x, top_k=args.top_k, use_abs=args.corr_abs)
    if adj_type == "identity":
        adj = identity
    elif adj_type == "static":
        if static_adj is None:
            print("Warning: static adjacency not found. Falling back to correlation adjacency.")
            adj = build_correlation_adj(train_set.data_x, top_k=args.top_k, use_abs=args.corr_abs)
        else:
            adj = static_adj
    elif adj_type in ["correlation", "corr"]:
        adj = corr_adj
    elif adj_type == "hybrid":
        if static_adj is None:
            print("Warning: hybrid has no static adjacency. Using correlation adjacency only.")
            adj = corr_adj
        else:
            if corr_adj is None:
                corr_adj = build_correlation_adj(train_set.data_x, top_k=args.top_k, use_abs=args.corr_abs)
            s = static_adj / (static_adj.max() + 1e-6)
            c = corr_adj / (corr_adj.max() + 1e-6)
            adj = args.hybrid_static_weight * s + (1.0 - args.hybrid_static_weight) * c
    else:
        raise ValueError("adj_type must be identity/static/correlation/hybrid")
    if adj.shape != (N, N):
        raise ValueError(f"Adjacency shape {tuple(adj.shape)} does not match variables N={N}")
    return adj.float()


def infer_dims(args, train_set):
    sample_x, sample_y, sample_x_mark, sample_y_mark = train_set[0]
    enc_in = int(train_set.data_x.shape[1])
    if args.features == "MS":
        c_out = 1
    elif args.features == "S":
        enc_in = 1
        c_out = 1
    else:
        c_out = enc_in
    args.enc_in = enc_in
    args.c_out = c_out
    args.time_dim = int(sample_x_mark.shape[-1]) if sample_x_mark is not None else 0
    return args


def select_target(batch_y: torch.Tensor, args) -> torch.Tensor:
    if args.features == "MS":
        return batch_y[:, -args.pred_len:, -args.c_out:]
    return batch_y[:, -args.pred_len:, :args.c_out]


class StreamingMetrics:
    def __init__(self):
        self.count = 0
        self.ae = 0.0
        self.se = 0.0
        self.ape = 0.0
        self.spe = 0.0
        self.true_sum = None
        self.true_sq_sum = None
        self.pred_sum = None
        self.pred_sq_sum = None
        self.cross_sum = None
        self.sample_count = 0

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        p = pred.detach().float().cpu()
        t = true.detach().float().cpu()
        diff = p - t
        n = diff.numel()
        self.count += n
        self.ae += diff.abs().sum().item()
        self.se += (diff * diff).sum().item()
        denom = t.abs().clamp(min=1e-5)
        self.ape += (diff.abs() / denom).sum().item()
        self.spe += ((diff / denom) ** 2).sum().item()
        # Sums over sample axis for approximate RSE/CORR.
        if self.true_sum is None:
            shape = t.shape[1:]
            self.true_sum = torch.zeros(shape)
            self.true_sq_sum = torch.zeros(shape)
            self.pred_sum = torch.zeros(shape)
            self.pred_sq_sum = torch.zeros(shape)
            self.cross_sum = torch.zeros(shape)
        self.sample_count += t.shape[0]
        self.true_sum += t.sum(dim=0)
        self.true_sq_sum += (t * t).sum(dim=0)
        self.pred_sum += p.sum(dim=0)
        self.pred_sq_sum += (p * p).sum(dim=0)
        self.cross_sum += (p * t).sum(dim=0)

    def result(self):
        mae = self.ae / max(1, self.count)
        mse = self.se / max(1, self.count)
        rmse = float(np.sqrt(mse))
        mape = self.ape / max(1, self.count)
        mspe = self.spe / max(1, self.count)
        true_mean = self.true_sum / max(1, self.sample_count)
        tss = (self.true_sq_sum - self.sample_count * true_mean * true_mean).clamp(min=1e-12).sum().item()
        rse = float(np.sqrt(self.se) / np.sqrt(tss)) if tss > 0 else float("nan")
        pred_mean = self.pred_sum / max(1, self.sample_count)
        cov = self.cross_sum - self.sample_count * pred_mean * true_mean
        var_p = (self.pred_sq_sum - self.sample_count * pred_mean * pred_mean).clamp(min=1e-12)
        var_t = (self.true_sq_sum - self.sample_count * true_mean * true_mean).clamp(min=1e-12)
        corr = (cov / torch.sqrt(var_p * var_t)).mean().item()
        return mae, mse, rmse, mape, mspe, rse, corr


def train_one_epoch(model, train_loader, optimizer, criterion, args, device, scaler) -> float:
    model.train()
    losses = []
    use_amp = bool(args.use_amp and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    for step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        batch_x = batch_x.float().to(device, non_blocking=True)
        batch_y = batch_y.float().to(device, non_blocking=True)
        batch_x_mark = batch_x_mark.float().to(device, non_blocking=True)
        batch_y_mark = batch_y_mark.float().to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch_x, batch_x_mark, None, batch_y_mark)
            target = select_target(batch_y, args)
            loss = criterion(outputs, target)
            loss_to_backward = loss / grad_accum_steps
        scaler.scale(loss_to_backward).backward()
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, data_loader, criterion, args, device, collect_arrays: bool = False):
    model.eval()
    use_amp = bool(args.use_amp and device.type == "cuda")
    losses = []
    preds = []
    trues = []
    stream = StreamingMetrics()
    for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
        batch_x = batch_x.float().to(device, non_blocking=True)
        batch_y = batch_y.float().to(device, non_blocking=True)
        batch_x_mark = batch_x_mark.float().to(device, non_blocking=True)
        batch_y_mark = batch_y_mark.float().to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch_x, batch_x_mark, None, batch_y_mark)
            target = select_target(batch_y, args)
            loss = criterion(outputs, target)
        losses.append(loss.item())
        stream.update(outputs, target)
        if collect_arrays:
            preds.append(outputs.detach().cpu().numpy())
            trues.append(target.detach().cpu().numpy())
    mean_loss = float(np.mean(losses))
    if collect_arrays:
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        metrics = metric(preds, trues)
        return mean_loss, metrics, preds, trues
    return mean_loss, stream.result(), None, None


def adjust_lr(optimizer, epoch: int, args):
    if args.lradj == "none":
        return
    if args.lradj == "type1":
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
    elif args.lradj == "cosine":
        lr = args.learning_rate * 0.5 * (1.0 + np.cos(np.pi * epoch / args.train_epochs))
    else:
        return
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"Learning rate adjusted to {lr:.8f}")


def main(args):
    set_seed(args.seed)
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
    if args.cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Using device: {device}")
    train_set, train_loader = data_provider(args, "train")
    val_set, val_loader = data_provider(args, "val")
    test_set, test_loader = data_provider(args, "test")
    args = infer_dims(args, train_set)
    print(f"Model dims | enc_in={args.enc_in}, c_out={args.c_out}, time_dim={args.time_dim}, seq_len={args.seq_len}, pred_len={args.pred_len}")

    static_adj = build_static_adj_for_training(args, train_set)
    model = GCN_mamba_TSForecast(args).to(device)
    model.set_static_adj(static_adj.to(device))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.3f}M")
    print(
        "Ablation switches | "
        f"bimamba={args.use_bimamba}, graph_adapter={args.use_graph_adapter}, "
        f"diffusion={args.use_graph_diffusion}, sna={args.use_sna}, "
        f"tanh_gate={args.use_tanh_gate}, orth_res={args.use_orth_res}, disable_ffn={args.disable_ffn}"
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp and device.type == "cuda"))

    setting = (
        f"{args.data}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_ly{args.layer_num}"
        f"_gh{args.graph_hops}_{args.adj_type}_topk{args.top_k}"
        f"_bi{int(args.use_bimamba)}_diff{int(args.use_graph_diffusion)}_sna{int(args.use_sna)}"
        f"_tanh{int(args.use_tanh_gate)}_opr{int(args.use_orth_res)}_seed{args.seed}"
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
    start_time = time.time()
    for epoch in range(1, args.train_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args, device, scaler)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, args, device, collect_arrays=False)
        print(f"Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f} | Time {time.time() - epoch_start:.2f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "static_adj": static_adj,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }, checkpoint_path)
            print(f"  Saved best checkpoint at epoch {epoch}.")
        else:
            patience_counter += 1
            print(f"  EarlyStopping counter: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
        adjust_lr(optimizer, epoch, args)

    print(f"\nTraining finished in {(time.time() - start_time) / 60:.2f} min.")
    print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")
    print("\nLoading best checkpoint and testing...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    collect_arrays = bool(args.save_preds)
    test_loss, test_metrics, preds, trues = evaluate(model, test_loader, criterion, args, device, collect_arrays=collect_arrays)
    mae, mse, rmse, mape, mspe, rse, corr = test_metrics
    print("\nTest result")
    print(f"Loss: {test_loss:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.6f}")
    print(f"MSPE: {mspe:.6f}")
    print(f"RSE : {rse:.6f}")
    print(f"CORR: {corr:.6f}")
    np.save(os.path.join(result_dir, "static_adj.npy"), static_adj.cpu().numpy())
    if collect_arrays:
        np.save(os.path.join(result_dir, "pred.npy"), preds)
        np.save(os.path.join(result_dir, "true.npy"), trues)
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


def get_args():
    parser = argparse.ArgumentParser(description="Ablation-ready SparseGraph-BiMamba for TS forecasting")
    parser.add_argument("--data", type=str, default="Traffic")
    parser.add_argument("--root_path", type=str, default="./dataset/traffic/")
    parser.add_argument("--data_path", type=str, default="traffic.csv")
    parser.add_argument("--adj_path", type=str, default="")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--value_channel", type=int, default=0)
    parser.add_argument("--auto_fix_split", type=str2bool, nargs="?", const=True, default=True)

    parser.add_argument("--adj_type", type=str, default="correlation", choices=["identity", "static", "correlation", "corr", "hybrid"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--graph_hops", type=int, default=1)
    parser.add_argument("--hop_decay", type=float, default=0.7)
    parser.add_argument("--corr_abs", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--hybrid_static_weight", type=float, default=0.7)

    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--layer_num", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--mamba_dropout", type=float, default=0.05)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--ffn_ratio", type=float, default=1.0)
    parser.add_argument("--ffn_chunk_nodes", type=int, default=64)
    parser.add_argument("--disable_ffn", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--alpha_res", type=float, default=0.03)
    parser.add_argument("--graph_scale_init", type=float, default=0.1)
    parser.add_argument("--graph_chunk_size", type=int, default=32768)
    parser.add_argument("--use_revin", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--revin_subtract_last", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_time_features", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--force_fallback_mamba", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--use_bimamba", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--use_graph_adapter", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--use_graph_diffusion", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--use_sna", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--use_tanh_gate", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--use_orth_res", type=str2bool, nargs="?", const=True, default=True)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lradj", type=str, default="cosine", choices=["none", "type1", "cosine"])
    parser.add_argument("--use_amp", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--drop_last", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--save_preds", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--results", type=str, default="./results/")
    parser.add_argument("--cuda_alloc_conf", type=str, default="expandable_segments:True")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
