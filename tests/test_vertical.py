import tomllib
from pathlib import Path

import pytest
from datasets import Dataset

from utils.partitioner_helper import PARTITIONERS


def _load_partition_config():
    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["partition"]


FEATURE_COLUMNS = [
    "Pclass", "Name", "Sex", "Age",
    "Siblings/Spouses Aboard", "Parents/Children Aboard", "Fare",
]


def _make_synthetic_titanic_dataset():
    n = 20
    data = {"Survived": [i % 2 for i in range(n)]}
    for col in FEATURE_COLUMNS:
        data[col] = list(range(n))
    return Dataset.from_dict(data)


@pytest.mark.parametrize("partitioner_type", ["vertical_even", "vertical_size"])
def test_vertical_partitioner_splits_columns(partitioner_type):
    cfg = _load_partition_config()
    dataset = _make_synthetic_titanic_dataset()

    partitioner = PARTITIONERS[partitioner_type](cfg, num_partitions=2)
    partitioner.dataset = dataset

    total_partitions = partitioner.num_partitions
    print(f"\n--- {partitioner_type} ---")

    seen_columns = []
    for i in range(total_partitions):
        part = partitioner.load_partition(i)
        print(f"Client {i}: {len(part.column_names)} columns -> {part.column_names}")
        assert len(part.column_names) > 0
        seen_columns.extend(part.column_names)

    assert sorted(seen_columns) == sorted(["Survived"] + FEATURE_COLUMNS)

    last_partition = partitioner.load_partition(total_partitions - 1)
    assert last_partition.column_names == ["Survived"]