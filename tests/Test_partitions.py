import tomllib
from collections import Counter
from pathlib import Path

import pytest
from datasets import Dataset

from utils.partitioner_helper import PARTITIONERS


def _load_partition_config():
    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)["partition"]


    cfg = dict(cfg)
    cfg["partition_sizes"] = [8, 12]
    cfg["min_partition_size"] = 1
    return cfg


def _make_synthetic_dataset():
    labels = [0] * 8 + [1] * 12
    modality = (["CT", "MRI"] * 10)
    return Dataset.from_dict({"label": labels, "modality": modality})


HORIZONTAL_PARTITIONERS = [k for k in PARTITIONERS if "vertical" not in k]


@pytest.mark.parametrize("partitioner_type", HORIZONTAL_PARTITIONERS)
def test_partitioner_runs_and_splits_data(partitioner_type):
    cfg = _load_partition_config()
    dataset = _make_synthetic_dataset()

    partitioner = PARTITIONERS[partitioner_type](cfg, num_partitions=2)
    partitioner.dataset = dataset

    print(f"\n--- {partitioner_type} (alpha={cfg.get('alpha')}) ---")

    total = 0
    for i in range(2):
        part = partitioner.load_partition(i)
        print(f"Client {i}: {len(part)} samples | labels={dict(Counter(part['label']))}")
        assert len(part) > 0
        total += len(part)

    assert total <= len(dataset)