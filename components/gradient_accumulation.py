import torch

from torch import nn


class BatchNormAccumulator:
    """
    Hook-based BatchNorm accumulator for gradient accumulation.
    
    - Accumulates `running_mean` and `running_var` across multiple accumulation steps.
    - Uses backward hooks to ensure correct timing of statistics updates.
    - Handles PyTorch BatchNorm's momentum updates properly.
    - Automatically removes hooks after training.
    """
    def __init__(self, model, num_accumulation_steps=1):
        self.model = model
        self.num_accumulation_steps = num_accumulation_steps
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers backward hooks for all BatchNorm layers in the model."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and module.track_running_stats:
                hook = module.register_backward_hook(self._backward_hook)
                self.hooks.append(hook)

    def _backward_hook(self, module, grad_input, grad_output):
        """Accumulates BatchNorm running statistics across accumulation steps."""
        if module.training and module.track_running_stats:
            if not hasattr(module, "accumulated_mean"):
                module.accumulated_mean = torch.zeros_like(module.running_mean)
                module.accumulated_var = torch.zeros_like(module.running_var)
                module.accumulated_count = 0

            # Accumulate statistics
            module.accumulated_mean += module.running_mean
            module.accumulated_var += module.running_var
            module.accumulated_count += 1

            # Apply updates only after `num_accumulation_steps`
            if module.accumulated_count == self.num_accumulation_steps:
                momentum = module.momentum or 0.1  # Default PyTorch BN momentum
                module.running_mean = (
                    module.running_mean * momentum +
                    (module.accumulated_mean / self.num_accumulation_steps) * (1 - momentum)
                )
                module.running_var = (
                    module.running_var * momentum +
                    (module.accumulated_var / self.num_accumulation_steps) * (1 - momentum)
                )

                # Reset accumulators
                del module.accumulated_mean
                del module.accumulated_var
                module.accumulated_count = 0

    def remove_hooks(self):
        """Removes all registered hooks after training."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        """Allows usage with `with` statements."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Automatically removes hooks after training when used in `with` statements."""
        self.remove_hooks()
