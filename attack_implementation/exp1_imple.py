# -*- coding: utf-8 -*-
import os
import hashlib
import itertools
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


def save_ieee_heatmap(
    M: np.ndarray,
    out_pdf: str,
    labels=None,
    figsize=(3.5, 3.2),
    fmt=".2f",
    annot_size=7,
    linewidths=0.2
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    if labels is None:
        labels = [str(i) for i in range(M.shape[0])]

    sns.heatmap(
        M,
        cmap="coolwarm",
        annot=True,
        fmt=fmt,
        annot_kws={"color": "white", "size": 20},
        cbar=False,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=linewidths,
        linecolor="white",
        ax=ax
    )

    ax.tick_params(axis="x", rotation=0, labelsize=7)
    ax.tick_params(axis="y", rotation=90, labelsize=7)

    plt.tight_layout(pad=0.01)

    out_dir = os.path.dirname(out_pdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_pdf)
    plt.close(fig)


def compute_cosine_similarity_matrix(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)

    required = {"user_id", "device_id", "event_type", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{csv_path}] missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    action_list = [
        "Read current temperature", "Read current humidity", "Read current lightlevel",
        "Turn on AC", "Turn off AC",
        "Turn on light", "Turn off light",
        "Turn on circulator fan", "Turn off circulator fan",
        "Read lock state", "Read door state", "Lock door", "Unlock door",
        "Turn on purifier", "Turn off purifier",
        "Turn on humidifier", "Turn off humidifier"
    ]

    top_devices = df["device_id"].value_counts().nlargest(7).index.tolist()

    features = []
    for user_id, g in df.groupby("user_id"):
        feat: Dict[str, float] = {"user_id": user_id}

        hour_dist = g["hour"].value_counts(normalize=True).reindex(range(24), fill_value=0)
        for h in range(24):
            feat[f"hour_{h}"] = float(hour_dist.loc[h])

        dev_dist = g["device_id"].value_counts(normalize=True)
        for dev in top_devices:
            feat[f"device_{dev}"] = float(dev_dist.get(dev, 0.0))

        evt_dist = g["event_type"].value_counts(normalize=True)
        for evt in action_list:
            feat[f"event_{evt}"] = float(evt_dist.get(evt, 0.0))

        features.append(feat)

    user_features = pd.DataFrame(features).fillna(0.0)
    feature_cols = [c for c in user_features.columns if c != "user_id"]
    X = user_features[feature_cols].to_numpy(dtype=float)

    sim = cosine_similarity(X)
    return sim


class CountingBloomFilter:
    def __init__(self, size: int, hash_count: int):
        self.size = size
        self.hash_count = hash_count
        self.counters = [0] * size

    def _hashes(self, item: str) -> List[int]:
        return [
            int(hashlib.sha256(f"{item}_{i}".encode()).hexdigest(), 16) % self.size
            for i in range(self.hash_count)
        ]

    def insert(self, item: str):
        for idx in self._hashes(item):
            self.counters[idx] += 1


def dice_coefficient(cbf1: CountingBloomFilter, cbf2: CountingBloomFilter) -> float:
    intersection = sum(min(a, b) for a, b in zip(cbf1.counters, cbf2.counters))
    total_a = sum(cbf1.counters)
    total_b = sum(cbf2.counters)
    return 1.0 if (total_a + total_b) == 0 else (2.0 * intersection / (total_a + total_b))


def compute_cbf_dice_similarity_matrix(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)

    if "hour" not in df.columns:
        if "timestamp" not in df.columns:
            raise ValueError(f"[{csv_path}] missing 'hour' and no 'timestamp' to derive it.")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour

    required = {"user_id", "device_id", "event_type", "hour"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{csv_path}] missing columns: {missing}")

    device_ids = [
        "02-202505201942-51413180", "02-202505201955-58136578",
        "B0E9FE892B71", "B08184D72E46", "D48C49DC31EA", "CB151A0062CD", "FEC966A1B4C1"
    ]
    device_encoding = {device_id: f"A{i + 1}" for i, device_id in enumerate(device_ids)}

    event_groups = {
        "B": ["Read current temperature", "Read current humidity", "Read current lightlevel"],
        "C": ["Lock door", "Unlock door", "Read lock state", "Read door state"],
        "D": ["Turn on AC", "Turn off AC"],
        "E": ["Turn on light", "Turn off light"],
        "F": ["Turn on circulator fan", "Turn off circulator fan"],
        "G": ["Turn on purifier", "Turn off purifier"],
        "H": ["Turn on humidifier", "Turn off humidifier"],
    }
    event_encoding = {
        event: f"{group}{i+1}"
        for group, events in event_groups.items()
        for i, event in enumerate(events)
    }

    def encode(device_id: str, event_type: str, hour: int) -> List[str]:
        if device_id not in device_encoding or event_type not in event_encoding:
            return []

        result = []
        device_code = device_encoding[device_id]
        event_code = event_encoding[event_type]
        mix_code = device_code[-1] + event_code[0]
        for offset in range(3):
            h = (int(hour) + offset) % 24
            result.append(f"_{h}{device_code}")
            result.append(f"_{h}{event_code}")
            result.append(f"_{h}{mix_code}")
        return result

    user_actions: Dict[str, List[tuple]] = {}
    for _, row in df.iterrows():
        user_actions.setdefault(row["user_id"], []).append(
            (row["device_id"], row["event_type"], int(row["hour"]))
        )

    cbf_dict: Dict[str, CountingBloomFilter] = {}
    for user_id, actions in user_actions.items():
        items: List[str] = []
        for device_id, event_type, hour in actions:
            items.extend(encode(device_id, event_type, hour))

        cbf = CountingBloomFilter(size=1024, hash_count=3)
        for it in items:
            cbf.insert(it)
        cbf_dict[user_id] = cbf

    user_ids = list(cbf_dict.keys())
    n = len(user_ids)
    sim_matrix = np.zeros((n, n), dtype=float)
    np.fill_diagonal(sim_matrix, 1.0)

    for i, j in itertools.combinations(range(n), 2):
        sim = dice_coefficient(cbf_dict[user_ids[i]], cbf_dict[user_ids[j]])
        sim_matrix[i, j] = sim_matrix[j, i] = sim

    return sim_matrix


def main():
    COS_CSV = "dataset/exp1_data_same_user.csv"
    CBF_CSV = "dataset/exp1_data_same_user.csv"

    OUT_COS_PDF = "exp_cos_ieee.pdf"
    OUT_CBF_PDF = "exp_cbf_ieee.pdf"

    FIGSIZE = (3.5, 3.2)
    ANNOT_SIZE = 7

    cos_sim = compute_cosine_similarity_matrix(COS_CSV)
    cos_labels = pd.read_csv(COS_CSV)["user_id"].drop_duplicates().astype(str).str[:5].tolist()

    save_ieee_heatmap(
        cos_sim,
        OUT_COS_PDF,
        labels=cos_labels,
        figsize=FIGSIZE,
        fmt=".2f",
        annot_size=ANNOT_SIZE,
        linewidths=0.2
    )

    cbf_sim = compute_cbf_dice_similarity_matrix(CBF_CSV)
    cbf_labels = pd.read_csv(CBF_CSV)["user_id"].drop_duplicates().astype(str).str[:5].tolist()

    save_ieee_heatmap(
        cbf_sim,
        OUT_CBF_PDF,
        labels=cbf_labels,
        figsize=FIGSIZE,
        fmt=".2f",
        annot_size=ANNOT_SIZE,
        linewidths=0.2
    )


if __name__ == "__main__":
    main()