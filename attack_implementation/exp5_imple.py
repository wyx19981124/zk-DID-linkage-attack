import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hashlib
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

FIGSIZE = (7, 4)
DPI = 300

def format_cell(x):
    if x < 0.05:
        return "0"
    elif np.isclose(x, 1):
        return "1"
    else:
        return f"{x:.1f}"

plt.rcParams.update({
    "figure.figsize": FIGSIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

CSV_PATH = 'dataset/exp5_data.csv'
df = pd.read_csv(CSV_PATH)

id_order = sorted(df['user_id'].astype(str).unique())
labels = [uid[:5] for uid in id_order]

device_ids = [
    '02-202505201942-51413180',
    '02-202505201955-58136578',
    'B0E9FE892B71', 'B08184D72E46',
    'D48C49DC31EA', 'CB151A0062CD', 'FEC966A1B4C1'
]
device_encoding = {device_id: f'A{i + 1}' for i, device_id in enumerate(device_ids)}

event_groups = {
    'B': ['Read current temperature', 'Read current humidity', 'Read current lightlevel'],
    'C': ['Lock door', 'Unlock door', 'Read lock state', 'Read door state'],
    'D': ['Turn on AC', 'Turn off AC'],
    'E': ['Turn on light', 'Turn off light'],
    'F': ['Turn on circulator fan', 'Turn off circulator fan'],
    'G': ['Turn on purifier', 'Turn off purifier'],
    'H': ['Turn on humidifier', 'Turn off humidifier']
}
event_encoding = {
    event: f'{group}{i+1}'
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
    return 1.0 if total_a + total_b == 0 else 2.0 * intersection / (total_a + total_b)

if 'hour' not in df.columns:
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
    else:
        raise ValueError("CSV must contain 'hour' or 'timestamp'.")

cbf_dict: Dict[str, CountingBloomFilter] = {}
for uid, g in df.groupby('user_id', sort=False):
    uid = str(uid)
    cbf = CountingBloomFilter(size=1024, hash_count=3)
    for _, r in g.iterrows():
        for t in encode(str(r['device_id']), str(r['event_type']), int(r['hour'])):
            cbf.insert(t)
    cbf_dict[uid] = cbf

n = len(id_order)
sim_matrix_cbf = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(n):
        sim_matrix_cbf[i, j] = dice_coefficient(cbf_dict[id_order[i]], cbf_dict[id_order[j]])

plt.figure(figsize=FIGSIZE)

annot_labels_cbf = np.vectorize(format_cell)(sim_matrix_cbf)

sns.heatmap(
    sim_matrix_cbf,
    xticklabels=labels,
    yticklabels=labels,
    cmap='coolwarm',
    annot=annot_labels_cbf,
    fmt="",
    annot_kws={"size": 6},
    square=False
)
plt.tight_layout()
plt.savefig("exp_cbf.pdf", bbox_inches='tight', pad_inches=0)
plt.close()

df2 = df.copy()
if 'timestamp' in df2.columns:
    df2['timestamp'] = pd.to_datetime(df2['timestamp'])
    df2['hour'] = df2['timestamp'].dt.hour

def extract_features(log_df: pd.DataFrame) -> pd.DataFrame:
    features = []
    action_list = list(event_encoding.keys())

    for uid, group in log_df.groupby('user_id', sort=False):
        uid = str(uid)
        feature = {'user_id': uid}

        hour_dist = group['hour'].value_counts(normalize=True).reindex(range(24), fill_value=0.0)
        for h in range(24):
            feature[f'hour_{h}'] = float(hour_dist[h])

        top_devices = log_df['device_id'].value_counts().nlargest(5).index
        device_dist = group['device_id'].value_counts(normalize=True)
        for dev in top_devices:
            feature[f'device_{dev}'] = float(device_dist.get(dev, 0.0))

        event_dist = group['event_type'].value_counts(normalize=True)
        for evt in action_list:
            feature[f'event_{evt}'] = float(event_dist.get(evt, 0.0))

        features.append(feature)

    return pd.DataFrame(features).fillna(0.0)

user_features = extract_features(df2)
user_features = user_features.set_index('user_id').loc[id_order].reset_index()

feature_cols = [col for col in user_features.columns if col != 'user_id']
scaler = MinMaxScaler()
X = scaler.fit_transform(user_features[feature_cols])

sim_matrix_cos = cosine_similarity(X)

plt.figure(figsize=FIGSIZE)

annot_labels_cos = np.vectorize(format_cell)(sim_matrix_cos)

sns.heatmap(
    sim_matrix_cos,
    xticklabels=labels,
    yticklabels=labels,
    cmap='coolwarm',
    annot=annot_labels_cos,
    fmt="",
    annot_kws={"size": 6},
    square=False
)
plt.tight_layout()
plt.savefig("exp_cos.pdf", bbox_inches='tight', pad_inches=0)
plt.close()