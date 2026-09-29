from datasets import load_dataset
from .base import BaseDatasetLoader


class TitanicLoader(BaseDatasetLoader):

    def load(self):
        dataset = load_dataset("julien-c/titanic-survival", split="train")
        dataset = dataset.rename_column("Survived", "label")

        titanic_dataset = dataset.train_test_split(
            test_size=0.2,
            seed=42,
        )

        return titanic_dataset