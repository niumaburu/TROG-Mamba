import argparse
import os
import numpy as np
import pandas as pd


def load_num_nodes_from_npz(npz_path: str) -> int:
    data = np.load(npz_path)

    # 优先使用常见 key
    if "data" in data.files:
        arr = data["data"]
    else:
        # 自动选择第一个维度合理的数组
        arr = None
        for key in data.files:
            candidate = data[key]
            if candidate.ndim >= 2:
                arr = candidate
                print(f"Use npz key '{key}' to infer num_nodes.")
                break
        if arr is None:
            raise ValueError(f"No valid array found in {npz_path}. Keys: {data.files}")

    print(f"Loaded npz array shape: {arr.shape}")

    # 常见 PEMS 格式：[T, N, C]
    if arr.ndim == 3:
        return int(arr.shape[1])

    # 如果是 [T, N]
    if arr.ndim == 2:
        return int(arr.shape[1])

    raise ValueError(f"Unsupported npz data shape: {arr.shape}")


def read_distance_csv(distance_csv: str) -> pd.DataFrame:
    """
    Return dataframe with columns: from, to, distance.
    Compatible with:
      from,to,cost
      from,to,distance
      src,dst,dist
      no-header csv
    """
    df = pd.read_csv(distance_csv)
    lower_cols = {str(c).strip().lower(): c for c in df.columns}

    src_candidates = ["from", "source", "src", "start", "i", "sensor1", "from_node"]
    dst_candidates = ["to", "target", "dst", "end", "j", "sensor2", "to_node"]
    dist_candidates = ["cost", "distance", "dist", "weight", "length"]

    src_col = next((lower_cols[c] for c in src_candidates if c in lower_cols), None)
    dst_col = next((lower_cols[c] for c in dst_candidates if c in lower_cols), None)
    dist_col = next((lower_cols[c] for c in dist_candidates if c in lower_cols), None)

    if src_col is not None and dst_col is not None and dist_col is not None:
        out = df[[src_col, dst_col, dist_col]].copy()
        out.columns = ["from", "to", "distance"]
    else:
        # 兼容无表头 csv
        out = pd.read_csv(distance_csv, header=None)
        if out.shape[1] < 3:
            raise ValueError(
                f"distance.csv should have at least 3 columns, got {out.shape[1]}"
            )
        out = out.iloc[:, :3].copy()
        out.columns = ["from", "to", "distance"]

    for col in ["from", "to", "distance"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["from", "to", "distance"])
    out["from"] = out["from"].astype(int)
    out["to"] = out["to"].astype(int)
    out["distance"] = out["distance"].astype(float)

    return out


def build_id_mapping(src: np.ndarray, dst: np.ndarray, num_nodes: int):
    """
    Map node ids in distance.csv to [0, num_nodes - 1].

    Most PEMS distance.csv files use 0-based indices directly.
    If using 1-based indices, this function converts them to 0-based.
    If using arbitrary sensor ids, it maps sorted ids to 0...N-1.
    """
    all_ids = np.unique(np.concatenate([src, dst]))
    min_id = int(all_ids.min())
    max_id = int(all_ids.max())

    # Case 1: already 0-based indices
    if min_id >= 0 and max_id < num_nodes:
        return {int(i): int(i) for i in all_ids}, "0-based direct index"

    # Case 2: 1-based indices
    if min_id >= 1 and max_id <= num_nodes:
        return {int(i): int(i) - 1 for i in all_ids}, "1-based index converted to 0-based"

    # Case 3: arbitrary sensor ids
    if len(all_ids) <= num_nodes:
        mapping = {int(node_id): idx for idx, node_id in enumerate(sorted(all_ids.tolist()))}
        return mapping, "arbitrary sensor id mapped by sorted order"

    raise ValueError(
        f"Cannot map node ids. num_nodes={num_nodes}, "
        f"min_id={min_id}, max_id={max_id}, unique_ids={len(all_ids)}"
    )


def build_adjacency(
    distance_csv: str,
    data_npz: str,
    out_path: str,
    sigma: str = "auto",
    threshold: float = 0.0,
    undirected: bool = True,
    add_self_loop: bool = True,
):
    num_nodes = load_num_nodes_from_npz(data_npz)
    df = read_distance_csv(distance_csv)

    src_raw = df["from"].values.astype(int)
    dst_raw = df["to"].values.astype(int)
    dist = df["distance"].values.astype(float)

    mapping, mapping_type = build_id_mapping(src_raw, dst_raw, num_nodes)
    print(f"Node id mapping: {mapping_type}")
    print(f"num_nodes inferred from npz: {num_nodes}")
    print(f"edges loaded from distance.csv: {len(df)}")

    if sigma == "auto":
        valid_dist = dist[np.isfinite(dist) & (dist > 0)]
        if len(valid_dist) == 0:
            sigma_value = 1.0
        else:
            sigma_value = float(valid_dist.std())
            if sigma_value <= 1e-12:
                sigma_value = float(valid_dist.mean())
            if sigma_value <= 1e-12:
                sigma_value = 1.0
    else:
        sigma_value = float(sigma)

    print(f"sigma = {sigma_value:.6f}")

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Gaussian kernel: larger distance -> smaller weight
    weights = np.exp(-np.square(dist / sigma_value)).astype(np.float32)

    if threshold > 0:
        weights[weights < threshold] = 0.0

    for s, t, w in zip(src_raw, dst_raw, weights):
        if w <= 0:
            continue

        if int(s) not in mapping or int(t) not in mapping:
            continue

        i = mapping[int(s)]
        j = mapping[int(t)]

        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            adj[i, j] = max(adj[i, j], float(w))
            if undirected:
                adj[j, i] = max(adj[j, i], float(w))

    if add_self_loop:
        np.fill_diagonal(adj, 1.0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, adj)

    nnz = int((adj > 0).sum())
    density = nnz / float(num_nodes * num_nodes)

    print(f"Saved adjacency to: {out_path}")
    print(f"adj shape: {adj.shape}")
    print(f"nonzero entries: {nnz}")
    print(f"density: {density:.6f}")
    print(f"min weight: {adj[adj > 0].min() if nnz > 0 else 0:.6f}")
    print(f"max weight: {adj.max():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_csv", type=str, required=True)
    parser.add_argument("--data_npz", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--sigma", type=str, default="auto",
                        help="'auto' or a float value. Gaussian kernel sigma.")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Set weights below threshold to zero. Recommend 0.0 when model already uses top_k.")
    parser.add_argument("--undirected", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--add_self_loop", type=lambda x: str(x).lower() == "true", default=True)

    args = parser.parse_args()

    build_adjacency(
        distance_csv=args.distance_csv,
        data_npz=args.data_npz,
        out_path=args.out,
        sigma=args.sigma,
        threshold=args.threshold,
        undirected=args.undirected,
        add_self_loop=args.add_self_loop,
    )


if __name__ == "__main__":
    main()