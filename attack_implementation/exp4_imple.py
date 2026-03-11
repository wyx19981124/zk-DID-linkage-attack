import os
import hashlib
import itertools
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
import seaborn as sns

DATA_PATH = 'dataset/exp4_data.csv'
TOP_N_DEVICES = 7
CBF_SIZE = 1024
CBF_HASH_COUNT = 3
THRESHOLDS = np.arange(0.0, 1.01, 0.01)

ACTION_LIST = [
    'Read current temperature','Read current humidity','Read current lightlevel',
    'Turn on AC','Turn off AC','Turn on light','Turn off light',
    'Turn on circulator fan','Turn off circulator fan',
    'Read lock state','Read door state','Lock door','Unlock door',
    'Turn on purifier','Turn off purifier','Turn on humidifier','Turn off humidifier'
]

EVENT_GROUPS = {
    'B': ['Read current temperature', 'Read current humidity', 'Read current lightlevel'],
    'C': ['Lock door', 'Unlock door', 'Read lock state', 'Read door state'],
    'D': ['Turn on AC', 'Turn off AC'],
    'E': ['Turn on light', 'Turn off light'],
    'F': ['Turn on circulator fan', 'Turn off circulator fan'],
    'G': ['Turn on purifier', 'Turn off purifier'],
    'H': ['Turn on humidifier', 'Turn off humidifier']
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

def normalize_bool_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.strip().str.lower()
         .map({'true': True, '1': True, 'yes': True, 'y': True,
               'false': False, '0': False, 'no': False, 'n': False})
    )

def mode_bool(s: pd.Series) -> bool:
    v = normalize_bool_series(s)
    m = v.mode(dropna=True)
    if not m.empty:
        return bool(m.iloc[0])
    return False

class CountingBloomFilter:
    def __init__(self, size: int, hash_count: int):
        self.size = size
        self.hash_count = hash_count
        self.counters = [0] * size

    def _hashes(self, item: str) -> List[int]:
        return [int(hashlib.sha256(f"{item}_{i}".encode()).hexdigest(), 16) % self.size
                for i in range(self.hash_count)]

    def insert(self, item: str):
        for idx in self._hashes(item):
            self.counters[idx] += 1

def dice_coefficient(cbf1: CountingBloomFilter, cbf2: CountingBloomFilter) -> float:
    intersection = sum(min(a, b) for a, b in zip(cbf1.counters, cbf2.counters))
    total_a = sum(cbf1.counters)
    total_b = sum(cbf2.counters)
    return 1.0 if total_a + total_b == 0 else 2.0 * intersection / (total_a + total_b)

def _confmat_for_threshold(sim_values, y_true, thr):
    y_pred = sim_values >= thr
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    return tn, fp, fn, tp

def sweep_metric(sim_values: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray, metric: str) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        tn, fp, fn, tp = _confmat_for_threshold(sim_values, y_true, thr)
        if metric == 'accuracy':
            total = tp + tn + fp + fn
            val = (tp + tn) / total if total else 0.0
        elif metric == 'precision':
            val = tp / (tp + fp) if (tp + fp) else 0.0
        elif metric == 'recall':
            val = tp / (tp + fn) if (tp + fn) else 0.0
        elif metric == 'f1':
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            val  = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        else:
            raise ValueError("metric must be one of: accuracy, precision, recall, f1")
        rows.append((thr, val, tp, tn, fp, fn))
    return pd.DataFrame(rows, columns=['threshold', metric, 'TP', 'TN', 'FP', 'FN'])

def plot_ieee_threshold_curve(thresholds, y_cos, y_cbf, ylabel, save_path, markevery=10):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ax.plot(thresholds, y_cos, marker='o', markevery=markevery, label='Cosine', alpha=0.95)
    ax.plot(thresholds, y_cbf, marker='s', markevery=markevery, label='CBF + Dice', alpha=0.95)

    ax.set_xlabel(r'$p$ value (threshold)')
    ax.set_ylabel(ylabel)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(True, which='major', linewidth=0.4, alpha=0.35)
    ax.grid(True, which='minor', linewidth=0.2, alpha=0.20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    bi_cos = int(np.argmax(y_cos))
    bi_cbf = int(np.argmax(y_cbf))
    thr_cos, val_cos = float(thresholds[bi_cos]), float(y_cos[bi_cos])
    thr_cbf, val_cbf = float(thresholds[bi_cbf]), float(y_cbf[bi_cbf])

    ax.axvline(thr_cos, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axvline(thr_cbf, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.scatter([thr_cos], [val_cos], s=18, zorder=3)
    ax.scatter([thr_cbf], [val_cbf], s=18, zorder=3)

    ax.annotate(
        f'{thr_cos:.2f}\n{val_cos:.1%}',
        xy=(thr_cos, val_cos),
        xytext=(min(1.0, thr_cos + 0.06), min(0.98, val_cos + 0.08)),
        arrowprops=dict(arrowstyle='->', lw=0.6),
        fontsize=12
    )

    ax.annotate(
        f'{thr_cbf:.2f}\n{val_cbf:.1%}',
        xy=(thr_cbf, val_cbf),
        xytext=(max(0.0, thr_cbf - 0.28), max(0.02, val_cbf - 0.18)),
        arrowprops=dict(arrowstyle='->', lw=0.6),
        fontsize=12
    )

    ax.legend(frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

os.makedirs('result', exist_ok=True)

df = pd.read_csv(DATA_PATH)
if 'hour' not in df.columns:
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
    else:
        raise ValueError("Input CSV must contain 'hour' or 'timestamp'.")

baseline_id = df['user_id'].iloc[0]

gt_user = (
    df.groupby('user_id')['ground_truth']
      .agg(mode_bool)
      .reset_index()
      .rename(columns={'ground_truth': 'ground_truth_user'})
)

all_user_ids = list(df['user_id'].drop_duplicates())
if all_user_ids[0] != baseline_id:
    all_user_ids = [baseline_id] + [u for u in all_user_ids if u != baseline_id]
candidates = all_user_ids[1:1001]

global_top_devices = df['device_id'].value_counts().nlargest(TOP_N_DEVICES).index.tolist()

def extract_features(log_df: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for uid, g in log_df.groupby('user_id'):
        row = {'user_id': uid}
        hour_dist = g['hour'].value_counts(normalize=True).reindex(range(24), fill_value=0.0)
        for h in range(24):
            row[f'hour_{h}'] = float(hour_dist[h])
        dev_dist = g['device_id'].value_counts(normalize=True)
        for dev in global_top_devices:
            row[f'device_{dev}'] = float(dev_dist.get(dev, 0.0))
        evt_dist = g['event_type'].value_counts(normalize=True)
        for evt in ACTION_LIST:
            row[f'event_{evt}'] = float(evt_dist.get(evt, 0.0))
        feats.append(row)
    return pd.DataFrame(feats).fillna(0.0)

user_features = extract_features(df)
feat_cols = [c for c in user_features.columns if c != 'user_id']
X = user_features.set_index('user_id')[feat_cols]

base_vec = X.loc[baseline_id].values.reshape(1, -1)
cos_rows = []
for uid in candidates:
    if uid not in X.index:
        continue
    sim = float(cosine_similarity(base_vec, X.loc[uid].values.reshape(1, -1))[0, 0])
    cos_rows.append((baseline_id, uid, sim))

cos_df = pd.DataFrame(cos_rows, columns=['baseline_id', 'user_id', 'similarity_cosine'])
cos_eval = cos_df.merge(gt_user, on='user_id', how='left')
cos_eval['ground_truth_user'] = cos_eval['ground_truth_user'].fillna(False)

event_encoding: Dict[str, str] = {}
for group, evs in EVENT_GROUPS.items():
    for i, ev in enumerate(evs, start=1):
        event_encoding[ev] = f'{group}{i}'
unknown_events = [ev for ev in df['event_type'].unique() if ev not in event_encoding]
if unknown_events:
    for i, ev in enumerate(sorted(unknown_events), start=1):
        event_encoding[ev] = f'Z{i}'

unique_devices = sorted(df['device_id'].astype(str).unique())
device_encoding = {dev: f'A{i+1}' for i, dev in enumerate(unique_devices)}

def encode(device_id: str, event_type: str, hour: int) -> List[str]:
    res = []
    dev_code = device_encoding.get(device_id)
    evt_code = event_encoding.get(event_type)
    if dev_code is None or evt_code is None:
        return res
    mix_code = dev_code[-1] + evt_code[0]
    for off in range(3):
        h = (int(hour) + off) % 24
        res.append(f"_{h}{dev_code}")
        res.append(f"_{h}{evt_code}")
        res.append(f"_{h}{mix_code}")
    return res

cbf_dict: Dict[str, CountingBloomFilter] = {}
for uid, g in df.groupby('user_id'):
    cbf = CountingBloomFilter(size=CBF_SIZE, hash_count=CBF_HASH_COUNT)
    for _, r in g.iterrows():
        toks = encode(str(r['device_id']), str(r['event_type']), int(r['hour']))
        for t in toks:
            cbf.insert(t)
    cbf_dict[uid] = cbf

cbf_rows = []
base_cbf = cbf_dict[baseline_id]
for uid in candidates:
    if uid not in cbf_dict:
        continue
    sim = dice_coefficient(base_cbf, cbf_dict[uid])
    cbf_rows.append((baseline_id, uid, sim))
cbf_df = pd.DataFrame(cbf_rows, columns=['baseline_id', 'user_id', 'similarity_cbf'])
cbf_eval = cbf_df.merge(gt_user, on='user_id', how='left')
cbf_eval['ground_truth_user'] = cbf_eval['ground_truth_user'].fillna(False)

y_true_cos = cos_eval['ground_truth_user'].astype(bool).values
sims_cos   = cos_eval['similarity_cosine'].values
y_true_cbf = cbf_eval['ground_truth_user'].astype(bool).values
sims_cbf   = cbf_eval['similarity_cbf'].values

acc_cos_df = sweep_metric(sims_cos, y_true_cos, THRESHOLDS, 'accuracy')
acc_cbf_df = sweep_metric(sims_cbf, y_true_cbf, THRESHOLDS, 'accuracy')
acc_df = acc_cos_df[['threshold','accuracy']].rename(columns={'accuracy':'accuracy_cosine'}).merge(
    acc_cbf_df[['threshold','accuracy']].rename(columns={'accuracy':'accuracy_cbf'}),
    on='threshold', how='inner'
)
acc_df.to_csv('result/threshold_metrics_compare.csv', index=False)
plot_ieee_threshold_curve(acc_df['threshold'].values, acc_df['accuracy_cosine'].values, acc_df['accuracy_cbf'].values,
                          ylabel='Accuracy', save_path='result/accuracy_vs_threshold_clean.pdf', markevery=10)

prec_cos_df = sweep_metric(sims_cos, y_true_cos, THRESHOLDS, 'precision')
prec_cbf_df = sweep_metric(sims_cbf, y_true_cbf, THRESHOLDS, 'precision')
prec_df = prec_cos_df[['threshold','precision']].rename(columns={'precision':'precision_cosine'}).merge(
    prec_cbf_df[['threshold','precision']].rename(columns={'precision':'precision_cbf'}),
    on='threshold', how='inner'
)
prec_df.to_csv('result/precision_metrics_compare.csv', index=False)
plot_ieee_threshold_curve(prec_df['threshold'].values, prec_df['precision_cosine'].values, prec_df['precision_cbf'].values,
                          ylabel='Precision', save_path='result/precision_vs_threshold_clean.pdf', markevery=10)

rec_cos_df = sweep_metric(sims_cos, y_true_cos, THRESHOLDS, 'recall')
rec_cbf_df = sweep_metric(sims_cbf, y_true_cbf, THRESHOLDS, 'recall')
rec_df = rec_cos_df[['threshold','recall']].rename(columns={'recall':'recall_cosine'}).merge(
    rec_cbf_df[['threshold','recall']].rename(columns={'recall':'recall_cbf'}),
    on='threshold', how='inner'
)
rec_df.to_csv('result/recall_metrics_compare.csv', index=False)
plot_ieee_threshold_curve(rec_df['threshold'].values, rec_df['recall_cosine'].values, rec_df['recall_cbf'].values,
                          ylabel='Recall', save_path='result/recall_vs_threshold_clean.pdf', markevery=10)

f1_cos_df = sweep_metric(sims_cos, y_true_cos, THRESHOLDS, 'f1')
f1_cbf_df = sweep_metric(sims_cbf, y_true_cbf, THRESHOLDS, 'f1')
f1_df = f1_cos_df[['threshold','f1']].rename(columns={'f1':'f1_cosine'}).merge(
    f1_cbf_df[['threshold','f1']].rename(columns={'f1':'f1_cbf'}),
    on='threshold', how='inner'
)
f1_df.to_csv('result/f1_metrics_compare.csv', index=False)
plot_ieee_threshold_curve(f1_df['threshold'].values, f1_df['f1_cosine'].values, f1_df['f1_cbf'].values,
                          ylabel='F1-score', save_path='result/f1_vs_threshold_clean.pdf', markevery=10)

print("Saved in ./result:")
print(" - accuracy_vs_threshold_clean.pdf, threshold_metrics_compare.csv")
print(" - precision_vs_threshold_clean.pdf, precision_metrics_compare.csv")
print(" - recall_vs_threshold_clean.pdf, recall_metrics_compare.csv")
print(" - f1_vs_threshold_clean.pdf, f1_metrics_compare.csv")