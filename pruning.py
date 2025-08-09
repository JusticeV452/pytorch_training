import torch


def fine_grained_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    # Step 1: calculate the #zeros (please use round())
    num_zeros = round(num_elements * sparsity)
    # Step 2: calculate the importance of weight
    importance = tensor.abs()
    # Step 3: calculate the pruning threshold
    threshold, _ = importance.reshape(-1).float().kthvalue(num_zeros)
    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = importance > threshold

    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)

    return mask


class FineGrainedPruner:
    def __init__(self, model, default_sparsity, sparsity_dict=None):
        self.sparsity_dict = {} if sparsity_dict is None else sparsity_dict
        self.default_sparsity = default_sparsity
        self.masks = self.create_masks(model)

    @torch.no_grad()
    def prune(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @torch.no_grad()
    def create_masks(self, model):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1: # we only prune conv and fc weights
                sparsity = self.sparsity_dict.get(name, self.default_sparsity)
                masks[name] = fine_grained_prune(param, sparsity)
        return masks


class DummyPruner(FineGrainedPruner):
    def __init__(self, model, default_sparsity, sparsity_dict=None):
        self.sparsity_dict = {} if sparsity_dict is None else sparsity_dict
        self.default_sparsity = default_sparsity
        self.masks = {}

    @torch.no_grad()
    def prune(self, model):
        return

    @torch.no_grad()
    def create_masks(self, model):
        return {}
