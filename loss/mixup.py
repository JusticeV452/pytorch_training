import torch
import torch.nn.functional as F


def mixup_data(real, fake, alpha=0.4):
    """Applies MixUp to real and fake images"""
    batch_size = real.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(real.device, real.dtype)
    lam = lam.view(batch_size, 1, 1, 1)  # Reshape for broadcasting
    mixed = lam * real + (1 - lam) * fake
    return mixed, lam


def mixup_bce(pred, lam):
    real_labels = torch.ones(pred.shape[0], 1, device=pred.device, dtype=pred.dtype)
    fake_labels = torch.zeros(pred.shape[0], 1, device=pred.device, dtype=pred.dtype)
    real_loss = lam * F.binary_cross_entropy(pred, real_labels, reduction='none')
    fake_loss = (1 - lam) * F.binary_cross_entropy(pred, fake_labels, reduction='none')
    return (real_loss + fake_loss).mean()
