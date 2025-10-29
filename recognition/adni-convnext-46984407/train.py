from torch.utils.data import ConcatDataset, Subset

def count_class_weights_from_train_dataset(train_ds, device):
    """Compute per-class weights robustly across dataset/concat/subset wrappers."""
    def counts_from_base(base, indices):
        targets = getattr(base, "targets", None)
        if targets is None:
            targets = [y for _, y in base.samples]
        n_classes = len(base.classes)
        counts = [0] * n_classes
        for i in indices:
            counts[targets[i]] += 1
        return counts

    if isinstance(train_ds, ConcatDataset):
        base_to_idxset = {}
        for sub in train_ds.datasets:
            base = getattr(sub, "base", None)
            idxs = getattr(sub, "indices", None)
            if base is None or idxs is None: continue
            if base not in base_to_idxset: base_to_idxset[base] = set()
            base_to_idxset[base].update(idxs)
        total_counts = None
        for base, idxset in base_to_idxset.items():
            c = counts_from_base(base, list(idxset))
            total_counts = c if total_counts is None else [a+b for a,b in zip(total_counts,c)]
        total = sum(total_counts)
        weights = [total/(c if c>0 else 1) for c in total_counts]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    if isinstance(train_ds, Subset):
        base, idxs = train_ds.dataset, train_ds.indices
        counts = counts_from_base(base, idxs)
        total = sum(counts)
        weights = [total/(c if c>0 else 1) for c in counts]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    targets = getattr(train_ds, "targets", None)
    if targets is None:
        targets = [y for _, y in train_ds.samples]
    n_classes = len(train_ds.classes)
    counts = [0]*n_classes
    for y in targets: counts[y]+=1
    total = sum(counts)
    weights = [total/(c if c>0 else 1) for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)
