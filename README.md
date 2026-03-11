# zk-DID-linkage-attack

An attack method to compromise **record linkability** and **identity linkability** in blockchain-based IoT systems.

## Overview

This repository contains code and datasets for reproducing a set of linkage-attack experiments on blockchain-based IoT interaction data. The project includes:

- Attack methods based on feature-distribution similarity and privacy-preserving record linkage style matching.
- Attack implementations for multiple experiments.
- Datasets used in the experiments.

The repository currently contains three main folders:

- `attack_method/`
- `attack_implementation/`
- `dataset/`

and also includes a top-level `requirements.txt`.

---

## Cosine similarity-based methods

These methods represent each user with behavioral distribution features including:

- hourly activity distribution
- device usage distribution
- event-type distribution

User similarity is then computed using **cosine similarity**.

Relevant files:

```
attack_method/cos-based_method.py
attack_method/cos-based_method(1-to-1).py
```

---

## PPRL-based methods

These methods encode device-event-time behaviors into symbolic tokens and insert them into a **Counting Bloom Filter (CBF)**.

Similarity is then computed using overlap-style metrics such as:

- Dice coefficient
- Tversky index

Relevant files:

```
attack_method/PPRL-based_method.py
attack_method/PPRL-based_method(tversky).py
```

---

## Implementations

The `attack_implementation/` folder contains experiment scripts corresponding to different experimental scenarios:

```
exp1_imple.py
exp2_imple.py
exp3_imple.py
exp4_imple.py
exp5_imple.py
```

These scripts provide runnable implementations for evaluating the linkage attacks under different datasets and settings.

---

## Datasets

The `dataset/` directory contains CSV files used in the experiments:

```
exp1_data_diff_user.csv
exp1_data_same_user.csv
exp2_data.csv
exp3_acl.csv
exp3_data.csv
exp4_data.csv
exp5_data.csv
```

Each dataset contains IoT interaction logs including fields such as:

- `user_id`
- `device_id`
- `event_type`
- `timestamp`
- `hour`

Please select the CSV file according to your experimental setting.

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```
