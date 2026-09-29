from collections import Counter
from task import get_dataset
from utils.partitioner_helper import get_partitioner

ds = get_dataset()["train"]
p = get_partitioner(2)
p.dataset = ds
for i in range(2):
    part = p.load_partition(i)
    print(i, len(part), dict(Counter(part["label"])), dict(Counter(part["modality"])))
    