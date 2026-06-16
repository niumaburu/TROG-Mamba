"""Robust dataloaders for CSV LTSF datasets and PEMS NPZ datasets.

Returned sample:
    seq_x:      [seq_len, C]
    seq_y:      [label_len + pred_len, C]
    seq_x_mark: [seq_len, time_feature_dim]
    seq_y_mark: [label_len + pred_len, time_feature_dim]
"""

import os
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

try:
    from timefeatures import time_features
except Exception:
    from utils.timefeatures import time_features

warnings.filterwarnings("ignore")


def _safe_read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".tsv"]:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path, header=None)
    return pd.read_csv(path)


def _make_regular_dates(length: int, freq: str, start: str = "2020-01-01") -> pd.Series:
    return pd.Series(pd.date_range(start=start, periods=length, freq=freq))


def _has_parseable_date(series: pd.Series) -> bool:
    try:
        parsed = pd.to_datetime(series.iloc[: min(50, len(series))], errors="coerce")
        return parsed.notna().mean() > 0.8
    except Exception:
        return False


def _prepare_dataframe(df_raw: pd.DataFrame, target: str, freq: str):
    df_raw = df_raw.copy()

    if "date" in df_raw.columns:
        date_col = "date"
    elif _has_parseable_date(df_raw.iloc[:, 0]):
        date_col = df_raw.columns[0]
        df_raw = df_raw.rename(columns={date_col: "date"})
        date_col = "date"
    else:
        df_raw.insert(0, "date", _make_regular_dates(len(df_raw), freq=freq))
        date_col = "date"

    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
    if df_raw[date_col].isna().any():
        df_raw[date_col] = _make_regular_dates(len(df_raw), freq=freq)

    numeric_cols = []
    for col in df_raw.columns:
        if col == "date":
            continue
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        if df_raw[col].notna().any():
            numeric_cols.append(col)

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset.")

    df_raw[numeric_cols] = df_raw[numeric_cols].interpolate(limit_direction="both").fillna(0.0)

    if target not in numeric_cols:
        target = numeric_cols[-1]
        print(f"Warning: target column not found. Use last numeric column '{target}' as target.")

    cols = [c for c in numeric_cols if c != target]
    df_raw = df_raw[["date"] + cols + [target]]
    return df_raw, target


def _build_time_features(df_stamp: pd.DataFrame, timeenc: int, freq: str):
    df_stamp = df_stamp.copy()
    df_stamp["date"] = pd.to_datetime(df_stamp["date"])
    if timeenc == 0:
        df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
        if str(freq).lower() in ["t", "min", "1min", "5min", "15min", "5t", "15t"]:
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
        data_stamp = df_stamp.drop(["date"], axis=1).values
    else:
        data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
    return data_stamp.astype(np.float32)


def _ratio_split_borders(
    total_len: int,
    seq_len: int,
    pred_len: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    auto_fix_split: bool = True,
):
    """Build chronological train/val/test borders.

    Important for LTSF windowing:
    - train samples = num_train - seq_len - pred_len + 1
    - val samples   = num_val - pred_len + 1, because validation has seq_len
      look-back overlap from the end of train
    - test samples  = num_test - pred_len + 1

    Therefore val/test must each contain at least pred_len points. The original
    0.7/0.1/0.2 split may fail on short CSV files, e.g. if total_len is only
    about 500 and pred_len=96. When auto_fix_split=True, val/test are expanded
    to pred_len points if possible, borrowing points from train.
    """
    total_len = int(total_len)
    seq_len = int(seq_len)
    pred_len = int(pred_len)
    min_train_len = seq_len + pred_len
    min_eval_len = pred_len
    min_total = min_train_len + 2 * min_eval_len

    if total_len < min_total:
        raise ValueError(
            f"Dataset is too short for seq_len={seq_len}, pred_len={pred_len}. "
            f"Need at least {min_total} rows to keep train/val/test non-empty, "
            f"but got total_len={total_len}. Reduce seq_len/pred_len or use a longer file."
        )

    num_train = int(total_len * train_ratio)
    num_test = int(total_len * test_ratio)
    num_val = total_len - num_train - num_test

    if auto_fix_split and (num_train < min_train_len or num_val < min_eval_len or num_test < min_eval_len):
        old = (num_train, num_val, num_test)
        num_val = max(num_val, min_eval_len)
        num_test = max(num_test, min_eval_len)
        num_train = total_len - num_val - num_test
        if num_train < min_train_len:
            # Last resort: keep minimum val/test and put the rest in train.
            num_val = min_eval_len
            num_test = min_eval_len
            num_train = total_len - num_val - num_test
        print(
            "Warning: split was automatically adjusted because at least one "
            f"segment was too short for pred_len={pred_len}. "
            f"old train/val/test={old}, new train/val/test={(num_train, num_val, num_test)}."
        )

    if num_train < min_train_len:
        raise ValueError(
            f"Training segment too short: train={num_train}, need at least "
            f"seq_len + pred_len = {min_train_len}. Adjust ratios or reduce lengths."
        )
    if num_val < min_eval_len or num_test < min_eval_len:
        raise ValueError(
            f"Validation/test segment too short: val={num_val}, test={num_test}, "
            f"each must be >= pred_len={pred_len}. Use --auto_fix_split true, "
            "increase val_ratio/test_ratio, or reduce pred_len."
        )

    border1s = [0, num_train - seq_len, num_train + num_val - seq_len]
    border2s = [num_train, num_train + num_val, total_len]
    return border1s, border2s, num_train, num_val, num_test


class _BaseTimeSeriesDataset(Dataset):
    def __init__(self, root_path: str, flag: str = "train", size: Optional[Sequence[int]] = None,
                 features: str = "M", data_path: str = "data.csv", target: str = "OT",
                 scale: bool = True, timeenc: int = 0, freq: str = "h",
                 train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2,
                 value_channel: int = 0, auto_fix_split: bool = True):
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 48, 96
        else:
            self.seq_len, self.label_len, self.pred_len = int(size[0]), int(size[1]), int(size[2])
        assert flag in ["train", "val", "test", "pred"]
        self.flag = flag
        self.set_type = {"train": 0, "val": 1, "test": 2}.get(flag, 0)
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.value_channel = int(value_channel)
        self.auto_fix_split = bool(auto_fix_split)
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
        self.scaler = StandardScaler()
        self.data_x = None
        self.data_y = None
        self.data_stamp = None
        self.raw_feature_columns = None
        self.__read_data__()

    def __read_data__(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # Return a non-negative integer so Python/DataLoader never raises the
        # cryptic error "__len__() should return >= 0". data_provider will
        # raise a clearer message if the dataset has no valid windows.
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(_BaseTimeSeriesDataset):
    """Generic CSV/TXT loader for Traffic, Electricity, Solar-Energy, Weather, etc."""
    def __read_data__(self):
        path = os.path.join(self.root_path, self.data_path)
        df_raw = _safe_read_table(path)
        df_raw, self.target = _prepare_dataframe(df_raw, self.target, self.freq)

        border1s, border2s, _, _, _ = _ratio_split_borders(
            len(df_raw), self.seq_len, self.pred_len,
            self.train_ratio, self.val_ratio, self.test_ratio, self.auto_fix_split
        )
        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        if self.features in ["M", "MS"]:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features must be one of M, S, MS")

        self.raw_feature_columns = list(df_data.columns)
        values = df_data.values.astype(np.float32)
        if self.scale:
            train_data = values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            values = self.scaler.transform(values).astype(np.float32)

        df_stamp = df_raw[["date"]][border1:border2]
        data_stamp = _build_time_features(df_stamp, self.timeenc, self.freq)
        self.data_x = values[border1:border2]
        self.data_y = values[border1:border2]
        self.data_stamp = data_stamp


class Dataset_ETT_hour(Dataset_Custom):
    pass


class Dataset_ETT_minute(Dataset_Custom):
    pass


class Dataset_PEMS(_BaseTimeSeriesDataset):
    """PEMS NPZ loader.

    Supported array shapes:
        [T, N]
        [T, N, C]  -> choose value_channel, default 0
    The npz key is preferably 'data'; otherwise the first array is used.
    """
    def __read_data__(self):
        path = os.path.join(self.root_path, self.data_path)
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.lib.npyio.NpzFile):
            if "data" in obj.files:
                data = obj["data"]
            else:
                data = obj[obj.files[0]]
            if "timestamp" in obj.files:
                stamp_raw = obj["timestamp"]
            elif "time" in obj.files:
                stamp_raw = obj["time"]
            else:
                stamp_raw = None
        else:
            data = obj
            stamp_raw = None

        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 3:
            ch = min(max(self.value_channel, 0), data.shape[-1] - 1)
            data = data[:, :, ch]
        elif data.ndim != 2:
            raise ValueError(f"PEMS data must be [T,N] or [T,N,C], got {data.shape}")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        T, N = data.shape
        border1s, border2s, _, _, _ = _ratio_split_borders(
            T, self.seq_len, self.pred_len,
            self.train_ratio, self.val_ratio, self.test_ratio, self.auto_fix_split
        )
        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data).astype(np.float32)

        if stamp_raw is not None:
            try:
                dates = pd.to_datetime(stamp_raw)
            except Exception:
                dates = pd.date_range("2020-01-01", periods=T, freq=self.freq)
        else:
            dates = pd.date_range("2020-01-01", periods=T, freq=self.freq)
        df_stamp = pd.DataFrame({"date": dates[border1:border2]})
        data_stamp = _build_time_features(df_stamp, self.timeenc, self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.raw_feature_columns = [f"node_{i}" for i in range(N)]
