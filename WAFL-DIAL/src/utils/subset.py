import random
from collections import defaultdict

from torch.utils.data import Subset


def create_subsets(
    dataset, num_nodes, class_to_node, bias_ratio=0.9, seed=1, label_extractor=None
):
    random.seed(seed)
    class_indices = defaultdict(list)

    for idx, item in enumerate(dataset):
        if label_extractor:
            label = label_extractor(item)
        else:
            _, label = item
        class_indices[label].append(idx)

    for cls in class_indices:
        random.shuffle(class_indices[cls])

    node_indices = [[] for _ in range(num_nodes)]

    for cls, indices in class_indices.items():
        main_node = class_to_node[cls]
        total = len(indices)
        bias_count = int(total * bias_ratio)

        node_indices[main_node].extend(indices[:bias_count])

        rest = indices[bias_count:]
        other_nodes = [i for i in range(num_nodes) if i != main_node]
        for i, idx in enumerate(rest):
            node = other_nodes[i % len(other_nodes)]
            node_indices[node].append(idx)

    subsets = [Subset(dataset, sorted(idxs)) for idxs in node_indices]
    return subsets
