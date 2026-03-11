import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib
from typing import List
import itertools
import matplotlib

# Use Tk backend for environments requiring GUI backend
matplotlib.use('TkAgg')

# ------------------------------------------------------------------
# Visualization Configuration
# ------------------------------------------------------------------

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ------------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------------

# Load IoT log data
# Replace the CSV file with the dataset corresponding to your experiment.
df = pd.read_csv('dataset/exp1_data_same_user')

# ------------------------------------------------------------------
# Device Encoding
# ------------------------------------------------------------------
# Each physical device is mapped to a short symbolic ID (A1, A2, ...)
# This reduces token length and improves Bloom filter efficiency.

device_ids = [
    '02-202505201942-51413180',
    '02-202505201955-58136578',
    'B0E9FE892B71',
    'B08184D72E46',
    'D48C49DC31EA',
    'CB151A0062CD',
    'FEC966A1B4C1'
]

device_encoding = {
    device_id: f'A{i + 1}'
    for i, device_id in enumerate(device_ids)
}

# ------------------------------------------------------------------
# Event Encoding
# ------------------------------------------------------------------
# Events are grouped by functionality and encoded using a group letter
# followed by an index. For example:
# B1 = Read temperature
# B2 = Read humidity

event_groups = {
    'B': [
        'Read current temperature',
        'Read current humidity',
        'Read current lightlevel'
    ],
    'C': [
        'Lock door',
        'Unlock door',
        'Read lock state',
        'Read door state'
    ],
    'D': [
        'Turn on AC',
        'Turn off AC'
    ],
    'E': [
        'Turn on light',
        'Turn off light'
    ],
    'F': [
        'Turn on circulator fan',
        'Turn off circulator fan'
    ],
    'G': [
        'Turn on purifier',
        'Turn off purifier'
    ],
    'H': [
        'Turn on humidifier',
        'Turn off humidifier'
    ]
}

event_encoding = {
    event: f'{group}{i + 1}'
    for group, events in event_groups.items()
    for i, event in enumerate(events)
}

# ------------------------------------------------------------------
# Encoding Function
# ------------------------------------------------------------------


def encode(device_id: str, event_type: str, hour: int) -> List[str]:
    """
    Encode a single user action into multiple symbolic tokens.

    The encoding considers:
    - Device identifier
    - Event type
    - Temporal context (current hour + next 2 hours)

    Each action generates 3 token types:
        device token
        event token
        device-event mixed token

    Parameters
    ----------
    device_id : str
        Unique identifier of the device
    event_type : str
        Description of the event
    hour : int
        Hour of occurrence (0–23)

    Returns
    -------
    List[str]
        Encoded tokens for insertion into the Bloom filter
    """

    tokens = []

    device_code = device_encoding[device_id]
    event_code = event_encoding[event_type]

    # Mixed code combines device index and event group
    mix_code = device_code[-1] + event_code[0]

    # Temporal expansion (current hour + 2 subsequent hours)
    for offset in range(3):
        h = (hour + offset) % 24

        tokens.append(f"_{h}{device_code}")
        tokens.append(f"_{h}{event_code}")
        tokens.append(f"_{h}{mix_code}")

    return tokens


# ------------------------------------------------------------------
# Counting Bloom Filter Implementation
# ------------------------------------------------------------------


class CountingBloomFilter:
    """
    A simple Counting Bloom Filter implementation.

    Unlike standard Bloom filters, counters allow tracking
    frequency of inserted items, enabling similarity computation.
    """

    def __init__(self, size: int, hash_count: int):

        self.size = size
        self.hash_count = hash_count
        self.counters = [0] * size

    def _hashes(self, item: str) -> List[int]:
        """
        Generate multiple hash indices for an item using SHA256.
        """

        return [
            int(
                hashlib.sha256(f"{item}_{i}".encode()).hexdigest(),
                16
            ) % self.size
            for i in range(self.hash_count)
        ]

    def insert(self, item: str):
        """
        Insert an item into the Counting Bloom Filter.
        """

        for idx in self._hashes(item):
            self.counters[idx] += 1

    def dump_counters(self) -> List[int]:
        """
        Return the internal counter array.
        """

        return self.counters


# ------------------------------------------------------------------
# Similarity Metric
# ------------------------------------------------------------------


def dice_coefficient(cbf1: CountingBloomFilter,
                     cbf2: CountingBloomFilter) -> float:
    """
    Compute Dice similarity between two Counting Bloom Filters.

    Dice coefficient:
        2 * |A ∩ B| / (|A| + |B|)
    """

    intersection = sum(
        min(a, b) for a, b in zip(cbf1.counters, cbf2.counters)
    )

    total_a = sum(cbf1.counters)
    total_b = sum(cbf2.counters)

    if total_a + total_b == 0:
        return 1.0

    return 2 * intersection / (total_a + total_b)


# ------------------------------------------------------------------
# Build User Action Sequences
# ------------------------------------------------------------------

user_actions = {}

for _, row in df.iterrows():
    user_actions.setdefault(row['user_id'], []).append(
        (row['device_id'], row['event_type'], row['hour'])
    )

# ------------------------------------------------------------------
# Encode User Behavior
# ------------------------------------------------------------------

comparison_dict = {}

for user_id, actions in user_actions.items():

    encoded_tokens = []

    for device_id, event_type, hour in actions:
        encoded_tokens.extend(
            encode(device_id, event_type, hour)
        )

    comparison_dict[user_id] = encoded_tokens

# ------------------------------------------------------------------
# Build Counting Bloom Filters
# ------------------------------------------------------------------

cbf_dict = {}

for user_id, tokens in comparison_dict.items():

    cbf = CountingBloomFilter(size=1024, hash_count=3)

    for token in tokens:
        cbf.insert(token)

    cbf_dict[user_id] = cbf

# ------------------------------------------------------------------
# Compute Pairwise User Similarity Matrix
# ------------------------------------------------------------------

user_ids = list(cbf_dict.keys())
n = len(user_ids)

sim_matrix = np.zeros((n, n))

for i, j in itertools.combinations(range(n), 2):

    sim = dice_coefficient(
        cbf_dict[user_ids[i]],
        cbf_dict[user_ids[j]]
    )

    sim_matrix[i][j] = sim
    sim_matrix[j][i] = sim

np.fill_diagonal(sim_matrix, 1.0)

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

plt.figure(figsize=(10, 8))

labels = [uid[:5] for uid in user_ids]

sns.heatmap(
    sim_matrix,
    xticklabels=labels,
    yticklabels=labels,
    cmap='coolwarm',
    annot=True
)

plt.tight_layout()

# Save high-resolution PDF figure
plt.savefig("exp_cbf.pdf", bbox_inches='tight')

plt.show()