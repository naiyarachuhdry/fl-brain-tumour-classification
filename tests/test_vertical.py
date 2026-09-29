from datasets_loaders import create_dataset
from utils.partitioner_helper import get_partitioner

ds = create_dataset("titanic", "unused").load()
print(ds)

p = get_partitioner(3)
p.dataset = ds["train"]
for i in range(p.num_partitions):
    print(i, p.load_partition(i).column_names)