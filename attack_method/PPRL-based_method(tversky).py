import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib
from typing import List
import itertools
import matplotlib

# Use Tk backend for environments that require GUI rendering
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

# Load IoT interaction data
# Replace the CSV file with the dataset corresponding to your experiment.
df = pd.read_csv('dataset/exp-3_compre_exp3.csv')

# ------------------------------------------------------------------
# Device Encoding
# ------------------------------------------------------------------
# Each device is mapped to a short symbolic identifier (A1, A2, ...).
# This reduces token length and simplifies subsequent encoding.

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
# B1 = Read current temperature
# C1 = Lock door

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
        Hour of occurrence (0-23)

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

    # Temporal expansion: current hour and the following two hours
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

    Unlike a standard Bloom filter, this version maintains counters
    instead of binary values, allowing frequency-aware similarity
    computation between encoded user behavior profiles.
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

    def insert(self, item: str) -> None:
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
# Similarity Metrics
# ------------------------------------------------------------------


def jaccard_similarity(
    cbf1: CountingBloomFilter,
    cbf2: CountingBloomFilter
) -> float:
    """
    Compute Jaccard similarity between two Counting Bloom Filters.

    Jaccard similarity is defined as:
        intersection / union
    where:
        intersection = sum of element-wise minimum values
        union        = sum of element-wise maximum values
    """

    intersection = sum(min(a, b) for a, b in zip(cbf1.counters, cbf2.counters))
    union = sum(max(a, b) for a, b in zip(cbf1.counters, cbf2.counters))

    return intersection / union if union != 0 else 1.0


def tversky_index(
    cbf1: CountingBloomFilter,
    cbf2: CountingBloomFilter,
    alpha: float = 0.4,
    beta: float = 0.6
) -> float:
    """
    Compute the Tversky index between two Counting Bloom Filters.

    The Tversky index generalizes Jaccard similarity by assigning
    different weights to asymmetric differences between two sets.

    Parameters
    ----------
    cbf1 : CountingBloomFilter
        First Counting Bloom Filter
    cbf2 : CountingBloomFilter
        Second Counting Bloom Filter
    alpha : float, optional
        Weight for features present in cbf1 but not in cbf2
    beta : float, optional
        Weight for features present in cbf2 but not in cbf1

    Returns
    -------
    float
        Tversky similarity score
    """

    intersection = sum(min(a, b) for a, b in zip(cbf1.counters, cbf2.counters))
    diff_a = sum(max(a - b, 0) for a, b in zip(cbf1.counters, cbf2.counters))
    diff_b = sum(max(b - a, 0) for a, b in zip(cbf1.counters, cbf2.counters))

    denominator = intersection + alpha * diff_a + beta * diff_b
    return intersection / denominator if denominator != 0 else 1.0

# ------------------------------------------------------------------
# Build User Action Sequences
# ------------------------------------------------------------------

# Group all device-event-hour tuples by user
user_actions = {}

for _, row in df.iterrows():
    user_actions.setdefault(row['user_id'], []).append(
        (row['device_id'], row['event_type'], row['hour'])
    )

# ------------------------------------------------------------------
# Encode User Behavior
# ------------------------------------------------------------------

# Convert each user's action sequence into encoded symbolic tokens
comparison_dict = {}

for user_id, actions in user_actions.items():
    encoded_tokens = []

    for device_id, event_type, hour in actions:
        encoded_tokens.extend(encode(device_id, event_type, hour))

    comparison_dict[user_id] = encoded_tokens

# ------------------------------------------------------------------
# Build Counting Bloom Filters
# ------------------------------------------------------------------

# Construct one Counting Bloom Filter for each user
cbf_dict = {}

for user_id, items in comparison_dict.items():
    cbf = CountingBloomFilter(size=1024, hash_count=3)

    for item in items:
        cbf.insert(item)

    cbf_dict[user_id] = cbf

# ------------------------------------------------------------------
# Similarity Matrix Construction
# ------------------------------------------------------------------

user_ids = list(cbf_dict.keys())
n = len(user_ids)

jaccard_matrix = np.zeros((n, n))
tversky_matrix = np.zeros((n, n))

for i, j in itertools.combinations(range(n), 2):
    cbf_i = cbf_dict[user_ids[i]]
    cbf_j = cbf_dict[user_ids[j]]

    jac = jaccard_similarity(cbf_i, cbf_j)
    tve = tversky_index(cbf_i, cbf_j, alpha=0.4, beta=0.6)

    jaccard_matrix[i][j] = jaccard_matrix[j][i] = jac
    tversky_matrix[i][j] = tversky_matrix[j][i] = tve

np.fill_diagonal(jaccard_matrix, 1.0)
np.fill_diagonal(tversky_matrix, 1.0)

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

# Use shortened user IDs for better readability in the heatmaps
labels = [uid[:5] for uid in user_ids]

# Jaccard similarity heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    jaccard_matrix,
    xticklabels=labels,
    yticklabels=labels,
    cmap='YlGnBu',
    annot=True
)
plt.title("User Similarity via Jaccard")
plt.tight_layout()
plt.savefig("user_similarity_jaccard.png", bbox_inches='tight')

# Tversky similarity heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    tversky_matrix,
    xticklabels=labels,
    yticklabels=labels,
    cmap='YlOrRd',
    annot=True
)
plt.title("User Similarity via Tversky (alpha=0.4, beta=0.6)")
plt.tight_layout()
plt.savefig("user_similarity_tversky.png", bbox_inches='tight')

# Display figures
plt.show()