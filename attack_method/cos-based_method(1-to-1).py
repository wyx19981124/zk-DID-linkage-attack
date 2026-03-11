import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
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

# Load IoT log data
# Replace the CSV file with the dataset corresponding to your experiment.
df = pd.read_csv('dataset/exp1_data_same_user.csv')

# Convert timestamp column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract temporal information from timestamp
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.date

# ------------------------------------------------------------------
# Feature Extraction
# ------------------------------------------------------------------


def extract_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Extract user-level behavioral feature vectors from IoT logs.

    Each user is represented by a feature vector containing:
    - Hourly activity distribution (24 dimensions)
    - Top device usage distribution
    - Event type distribution

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input IoT log dataset

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to one user
        and each column corresponds to one behavioral feature
    """

    features = []

    # Predefined list of event types to ensure consistent feature dimensions
    action_list = [
        'Read current temperature',
        'Read current humidity',
        'Read current lightlevel',
        'Turn on AC',
        'Turn off AC',
        'Turn on light',
        'Turn off light',
        'Turn on circulator fan',
        'Turn off circulator fan',
        'Read lock state',
        'Read door state',
        'Lock door',
        'Unlock door',
        'Turn on purifier',
        'Turn off purifier',
        'Turn on humidifier',
        'Turn off humidifier'
    ]

    # Select the top 7 most frequently used devices across the entire dataset
    top_devices = dataframe['device_id'].value_counts().nlargest(7).index

    # Build one feature vector for each user
    for user_id, group in dataframe.groupby('user_id'):
        feature = {'user_id': user_id}

        # --------------------------------------------------------------
        # 1. Hour-of-day activity distribution
        # --------------------------------------------------------------
        # Compute normalized interaction frequency for each hour (0-23)
        hour_dist = (
            group['hour']
            .value_counts(normalize=True)
            .reindex(range(24), fill_value=0)
        )

        for h in range(24):
            feature[f'hour_{h}'] = hour_dist[h]

        # --------------------------------------------------------------
        # 2. Device usage distribution
        # --------------------------------------------------------------
        # Compute normalized usage frequency for the globally top devices
        device_dist = group['device_id'].value_counts(normalize=True)

        for dev in top_devices:
            feature[f'device_{dev}'] = device_dist.get(dev, 0)

        # --------------------------------------------------------------
        # 3. Event type distribution
        # --------------------------------------------------------------
        # Compute normalized occurrence frequency for each predefined event
        event_dist = group['event_type'].value_counts(normalize=True)

        for evt in action_list:
            feature[f'event_{evt}'] = event_dist.get(evt, 0)

        features.append(feature)

    feature_df = pd.DataFrame(features).fillna(0)
    return feature_df


user_features = extract_features(df)

# ------------------------------------------------------------------
# Feature Matrix Construction
# ------------------------------------------------------------------

# Exclude user_id column and directly use raw feature values
# No normalization is applied in this version.
feature_cols = [col for col in user_features.columns if col != 'user_id']
X = user_features[feature_cols].values

# ------------------------------------------------------------------
# Similarity Computation
# ------------------------------------------------------------------

# Compute pairwise cosine similarity between all user feature vectors
similarity_matrix = cosine_similarity(X)

# ------------------------------------------------------------------
# Similarity Heatmap Visualization
# ------------------------------------------------------------------

plt.figure(figsize=(10, 8))

# Use shortened user IDs for better readability in the heatmap
short_labels = [uid[:5] for uid in user_features['user_id']]

sns.heatmap(
    similarity_matrix,
    xticklabels=short_labels,
    yticklabels=short_labels,
    cmap='coolwarm',
    annot=True
)

plt.tight_layout()
plt.savefig("exp_cos.pdf", bbox_inches='tight')

# ------------------------------------------------------------------
# Top-N Similar User Retrieval
# ------------------------------------------------------------------

# For each user, identify the Top-N most similar users
top_n = 3
user_ids = user_features['user_id'].tolist()

for i, uid in enumerate(user_ids):
    sim_scores = list(enumerate(similarity_matrix[i]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the user itself and keep the top-N similar users
    top_similar = [
        (user_ids[j], score)
        for j, score in sim_scores[1:top_n + 1]
    ]

    print(f'User {uid} Top-{top_n} similar users: {top_similar}')

# ------------------------------------------------------------------
# Display Figure
# ------------------------------------------------------------------

plt.show()