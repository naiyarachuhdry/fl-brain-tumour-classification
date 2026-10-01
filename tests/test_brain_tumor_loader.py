from pathlib import Path

from datasets_loaders.brain_tumor_dataset import BrainTumorLoader

DATA_DIR = Path(r"D:\Dataset")


def test_dataset_length():
    loader = BrainTumorLoader(str(DATA_DIR))
    dataset = loader.load()

    total = len(dataset["train"]) + len(dataset["test"])
    assert total == 9618