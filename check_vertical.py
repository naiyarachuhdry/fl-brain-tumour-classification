from datasets import load_dataset
from utils.partitioner_helper import get_partitioner

ds = load_dataset("julien-c/titanic-survival", split="train")
print(ds.column_names)

p = get_partitioner(3)
p.dataset = ds
for i in range(p.num_partitions):
    print(i, p.load_partition(i).column_names)