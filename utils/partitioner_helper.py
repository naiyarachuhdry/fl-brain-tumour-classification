import tomllib
from pathlib import Path

import numpy as np
from flwr_datasets.partitioner import (
    IidPartitioner,
    DirichletPartitioner,
    ShardPartitioner,
    PathologicalPartitioner,
    InnerDirichletPartitioner,
    NaturalIdPartitioner,
    GroupedNaturalIdPartitioner,
    SizePartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
    DistributionPartitioner,
    VerticalEvenPartitioner,
    VerticalSizePartitioner,
)

PARTITIONERS = {
    "iid": lambda cfg, num_partitions: IidPartitioner(
        num_partitions=num_partitions,
    ),
    "dirichlet": lambda cfg, num_partitions: DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=cfg["alpha"],
        seed=cfg.get("seed", 42),
    ),
    "shard": lambda cfg, num_partitions: ShardPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        num_shards_per_partition=cfg["num_shards"],
        seed=cfg.get("seed", 42),
    ),
    "pathological": lambda cfg, num_partitions: PathologicalPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        num_classes_per_partition=cfg["num_classes"],
        class_assignment_mode=cfg["class_mode"],
        seed=cfg.get("seed", 42),
    ),
    "inner_dirichlet": lambda cfg, num_partitions: InnerDirichletPartitioner(
        partition_sizes=cfg["partition_sizes"],
        partition_by="label",
        alpha=cfg["alpha"],
        seed=cfg.get("seed", 42),
    ),
    "natural_id": lambda cfg, num_partitions: NaturalIdPartitioner(
        partition_by=cfg["natural_id_column"],
    ),
    "grouped_natural_id": lambda cfg, num_partitions: GroupedNaturalIdPartitioner(
        partition_by=cfg["natural_id_column"],
        group_size=cfg["group_size"],
    ),
    "size": lambda cfg, num_partitions: SizePartitioner(
        partition_sizes=cfg["partition_sizes"],
    ),
    "linear": lambda cfg, num_partitions: LinearPartitioner(
        num_partitions=num_partitions,
    ),
    "square": lambda cfg, num_partitions: SquarePartitioner(
        num_partitions=num_partitions,
    ),
    "exponential": lambda cfg, num_partitions: ExponentialPartitioner(
        num_partitions=num_partitions,
    ),
    "distribution": lambda cfg, num_partitions: DistributionPartitioner(
        distribution_array=np.array(cfg["distribution_array"]),
        num_partitions=num_partitions,
        num_unique_labels_per_partition=cfg["labels_per_partition"],
        partition_by="label",
        preassigned_num_samples_per_label=cfg["preassigned_per_label"],
        seed=cfg.get("seed", 42),
    ),
    "vertical_even": lambda cfg, num_partitions: VerticalEvenPartitioner(
        num_partitions=num_partitions,
        active_party_columns=cfg.get("active_party_column", "label"),
        active_party_columns_mode="create_as_last",
        seed=cfg.get("seed", 42),
    ),
    "vertical_size": lambda cfg, num_partitions: VerticalSizePartitioner(
        partition_sizes=cfg["vertical_sizes"],
        active_party_columns=cfg.get("active_party_column", "label"),
        active_party_columns_mode="create_as_last",
        seed=cfg.get("seed", 42),
    ),
}


def get_partitioner(num_partitions):

    config = {}

    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = BASE_DIR / "pyproject.toml"

    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    cfg = config["partition"]

    name = cfg["type"].lower()
    if name not in PARTITIONERS:
        raise ValueError(f"Unsupported partition type: {cfg['type']}")

    return PARTITIONERS[name](cfg, num_partitions)