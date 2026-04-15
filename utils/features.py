import torch

PROPKA_FEATURES = {
    'pka':         512,
    'desolvation': 513,
    'bb_hbond':    514,
    'sc_hbond':    515,
    'coulomb':     516,
    'combined':    517,
}


def apply_feature_exclusion(data_list, exclude_names):
    """Remove specified PROPKA feature columns from every Data.x tensor.

    Args:
        data_list: list of PyG Data objects with x of shape [L, 518]
        exclude_names: list of PROPKA feature names to drop, or
                       ['all_propka'] to drop all six

    Returns:
        (data_list, input_dim) with data_list mutated in-place
    """
    if not exclude_names:
        return data_list, 518

    exclude_indices = set()
    for name in exclude_names:
        if name == 'all_propka':
            exclude_indices.update(range(512, 518))
        elif name in PROPKA_FEATURES:
            exclude_indices.add(PROPKA_FEATURES[name])
        else:
            raise ValueError(f"Unknown PROPKA feature: {name}. "
                             f"Choose from {list(PROPKA_FEATURES.keys())} or 'all_propka'")

    keep = sorted(set(range(518)) - exclude_indices)
    keep_t = torch.tensor(keep, dtype=torch.long)

    for data in data_list:
        data.x = data.x[:, keep_t]

    input_dim = len(keep)
    excluded_str = ', '.join(sorted(exclude_names))
    print(f"[Ablation] Excluded features: {excluded_str} -> input_dim = {input_dim}")
    return data_list, input_dim
