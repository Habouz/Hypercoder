import numpy as np


def count_effective_nodes(W, b = None, threshold=1e-3):
    """
    Count effective neurons in a layer: those with any incoming weight or bias
    exceeding the threshold in magnitude.

    Parameters
    ----------
    W : np.ndarray, shape (n_nodes, n_inputs)
        Weight matrix of the layer.
    b : np.ndarray, shape (n_nodes,)
        Bias vector of the layer.
    threshold : float
        Values below this absolute threshold are considered ineffective.

    Returns
    -------
    int
        Number of effective neurons in the layer.
    """
    # For each neuron, check if any weight or its bias is above threshold
    weights_abs = np.abs(W)
    biases_abs = np.abs(b) if b is not None else np.zeros(W.shape[0])

    # A neuron is effective if it has any large weight or bias
    effective = (weights_abs >= threshold).any(axis=1) | (biases_abs >= threshold)
    effective_indices = np.where(effective)[0]  # Indices of effective nodes
    print(f"Effective node indices: {effective_indices}") if __name__ == "__main__" else None
    return int(effective.sum()), effective_indices


def find_layer_with_least_effective(layers, threshold=-1):
    """
    Determine which layer has the fewest effective neurons.

    Parameters
    ----------
    layers : list of tupl

        Each element is (W, b) for a layer, where W is the weight matrix
        and b is the bias vector (both numpy arrays).
    threshold : float
        Threshold for considering weights or biases as effective.

    Returns
    -------
    (layer_index, count) : tuple
        Zero-based index of the layer with the least effective neurons,
        and the number of effective neurons in that layer.
    """

    if threshold < 0:
        max_thres = max([max(np.abs(W).max(), np.abs(b).max()) if b is not None else np.abs(W).max() for W, b in layers])
        threshold = max_thres * 0.01
    counts = []
    effective_indices = []
    for idx, (W,b) in enumerate(layers):
        cnt, eff = count_effective_nodes(W,b,threshold)
        counts.append(cnt)
        effective_indices.append(eff)
        print(f"Layer {idx}: {cnt} effective nodes") if __name__ == "__main__" else None

    min_count = min(counts)
    min_idx = counts.index(min_count)
    eff = effective_indices[min_idx]
    print(f"Layer {min_idx} has the least effective nodes: {min_count}", end = '\n') if __name__ == "__main__" else None

    return min_idx, min_count, threshold, eff
