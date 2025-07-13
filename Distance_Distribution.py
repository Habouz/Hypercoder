import torch

def cartesian_product_gpu(tensor_list):
    device = tensor_list[0].device
    dtype = tensor_list[0].dtype
    sizes = [len(t) for t in tensor_list]
    n = len(tensor_list)

    # Total number of combinations
    total = torch.prod(torch.tensor(sizes, device=device))

    # Initialize output tensor
    out = torch.empty((total, n), device=device, dtype=dtype)

    repeats = 1
    for i in reversed(range(n)):
        t = tensor_list[i]
        t_expanded = t.repeat_interleave(repeats)
        tile_factor = total // (repeats * len(t))
        out[:, i] = t_expanded.tile(tile_factor)
        repeats *= len(t)

    return out

def all_path_list(first_node, last_node, weight_list, bias_list=None):
    device = weight_list[0].device
    sizes = [w.size(0) for w in weight_list] + [weight_list[-1].size(1)]
    grids = [torch.arange(s, device=device) for s in sizes]
    all_paths = torch.cartesian_prod(*grids)

    valid_mask = (all_paths[:, 0] == first_node) & (all_paths[:, -1] == last_node)
    selected_paths = all_paths[valid_mask]

    path_weights = []
    for path in selected_paths:
        weight = torch.tensor(1.0, device=device)
        for i in range(len(weight_list)):
            if bias_list is not None and bias_list[i] is not None:
                weight = weight * (torch.abs(weight_list[i][path[i], path[i + 1]]) + torch.abs(bias_list[i][path[i + 1]]))
        path_weights.append(weight)

    return selected_paths, torch.stack(path_weights) if path_weights else torch.tensor([], device=device)

def path_entropy(first_node, last_node, weight_list, bias_list=None):
    paths, weights = all_path_list(first_node, last_node, weight_list, bias_list)
    if weights.numel() == 0:
        return torch.tensor(0.0, device=weight_list[0].device, requires_grad=True)

    abs_weights = weights.abs()
    total_weight = abs_weights.sum()
    probs = abs_weights / (total_weight + 1e-9)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9))
    return entropy

def path_entropy_parallel(first_node, last_node, weight_list, bias_list=None, a=1, paths = None):
    device = weight_list[0].device

    if paths is None:
        sizes = [w.size(0) for w in weight_list] + [weight_list[-1].size(1)]
        grids = [torch.arange(s, device=device) for s in sizes]
        # all_paths = torch.cartesian_prod(*grids).to(device)
        all_paths = cartesian_product_gpu(grids)

        # Only keep paths from first_node to last_node
        valid_mask = (all_paths[:, 0] == first_node) & (all_paths[:, -1] == last_node)
        selected_paths = all_paths[valid_mask]
    else:
        selected_paths = paths

    if selected_paths.size(0) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    num_paths = selected_paths.size(0)
    num_layers = len(weight_list)

    weights = torch.ones(num_paths, device=device)

    for i in range(num_layers):
        src_idx = selected_paths[:, i]
        tgt_idx = selected_paths[:, i + 1]

        # Apply weight
        W = weight_list[i]
        # print(W.device)

        # Apply bias at current source node
        if bias_list is not None and bias_list[i] is not None:
            weights *= torch.abs(W[src_idx, tgt_idx]) + torch.abs(bias_list[i][tgt_idx])
        else:
            weights *= torch.abs(W[src_idx, tgt_idx])
    # print(f"device of weights {weights.device}")
    # print(f"device of W {W.device}")
    # print(f"src {src_idx.device} tgt {tgt_idx.device}")


    # Normalize and compute entropy
    abs_weights = weights.abs()
    total = abs_weights.sum()
    probs = abs_weights / (total + 1e-9)

    if a == 1:
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9))
    else:
        entropy = a / (1 - a) * torch.log2(torch.norm(probs, p=a))

    return entropy, selected_paths


if __name__ == "__main__":
    weight_matrices = [
        torch.tensor([[0, 1, 0],
                      [1, 2, 5],
                      [5, 2, 6]]),
        torch.tensor([[5, 1, 0],
                      [0, 1, 1],
                      [6, 9, 2]]), 
        torch.tensor([[4, 2, 0],
                      [6, 3, 1],
                      [1, 2, 1]])]
    
    bias_list = [
        torch.tensor([1.0, 0.1, 0.2]),
        torch.tensor([0.1, 0.2, 0.3]),
        torch.tensor([0.2, 0.3, 0.4])
    ]
    first_node = 0
    last_node = 2

    selected_paths, weight_list = all_path_list(first_node, last_node, weight_matrices, bias_list)
    print(f"Selected paths from {first_node} to {last_node}:")
    for path, weight in zip(selected_paths, weight_list):
        print(f"Path: {path.tolist()}, Product of weights: {weight.item()}")
    entropy = path_entropy(first_node, last_node, weight_matrices, bias_list)
    print(f"Entropy of paths from {first_node} to {last_node}: {entropy.item()}")
    entropy_parallel, paths = path_entropy_parallel(first_node, last_node, weight_matrices, bias_list)
    entropy_parallel2, _ = path_entropy_parallel(first_node, last_node, weight_matrices, bias_list, a=1, paths=paths)
    print(f"Parallel Entropy of paths from {first_node} to {last_node}: {entropy_parallel.item()}")
    print(f"Parallel Entropy with a=0.2 of paths from {first_node} to {last_node}: {entropy_parallel2}")
    # entropy_parallel_gpu = path_entropy_parallel_gpu(first_node, last_node, weight_matrices, bias_list)
    # print(f"Parallel GPU Entropy of paths from {first_node} to {last_node}: {entropy_parallel_gpu.item()}")