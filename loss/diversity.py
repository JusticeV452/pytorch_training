import torch
import torch.nn.functional as F

from torch import nn
from torchmetrics.image.kid import KernelInceptionDistance
from torch_fidelity import calculate_metrics
from utils import channel_width_flatten


def compute_kid(real_images, fake_images):
    """
    Compute Kernel Inception Distance (KID).
    Args:
        real_images (torch.Tensor): Real images (N, C, H, W)
        fake_images (torch.Tensor): Generated images (N, C, H, W)
    Returns:
        float: KID score
    """
    with torch.no_grad():
        kid = KernelInceptionDistance(subset_size=len(real_images)).to(real_images.device)
        kid.update((channel_width_flatten(real_images) * 255).to(torch.uint8), real=True)
        kid.update((channel_width_flatten(fake_images) * 255).to(torch.uint8), real=False)
        return kid.compute()[0].item()


def compute_fid(real_images, fake_images):
    """
    Compute FID score between real and generated images.
    Args:
        real_images (torch.Tensor): Real images (N, C, H, W)
        fake_images (torch.Tensor): Generated images (N, C, H, W)
    Returns:
        float: FID score
    """
    with torch.no_grad():
        metrics = calculate_metrics({"real": real_images[:, -3:], "fake": fake_images[:, -3:]}, cuda=True)
        return metrics["frechet_inception_distance"]


def orthogonality_loss(features):
    B, C, H, W = features.shape
    reshaped = features.view(B, C, -1)  # Flatten spatial dimensions
    gram_matrix = torch.bmm(reshaped, reshaped.transpose(1, 2))  # C x C similarity
    identity = torch.eye(C).to(features.device)
    loss = torch.mean((gram_matrix - identity) ** 2)  # Penalize non-diagonal values
    return loss


def chunked_orthogonality_loss(features, chunk_size=1):
    """
    Computes orthogonality loss in chunks to reduce memory overhead.
    
    Args:
        features (Tensor): Feature maps of shape (B, C, H, W).
        chunk_size (int): Number of channels to process at a time.

    Returns:
        loss (Tensor): Scalar diversity loss.
    """
    B, C, H, W = features.shape
    reshaped = features.view(B, C, -1)  # Flatten spatial

    total_loss = 0
    num_chunks = (C + chunk_size - 1) // chunk_size  # Number of chunks

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, C)
        gram_matrix = torch.bmm(reshaped[:, start:end, :], reshaped[:, start:end, :].transpose(1, 2))
        identity = torch.eye(end - start).to(features.device)
        total_loss += torch.mean((gram_matrix - identity) ** 2)

    return total_loss / num_chunks  # Normalize loss


def pooled_orthogonality_loss(features, reduce_dim=True, pool_size=64, num_chunks=2, norm=True):
    """
    Computes feature orthogonality loss efficiently to prevent OOM issues.
    
    Args:
        features (Tensor): Feature maps of shape (B, C, H, W).
        reduce_dim (bool): Whether to reduce spatial dimensions via pooling.
        pool_size (int): Pooling kernel size for spatial downsampling.

    Returns:
        loss (Tensor): Scalar diversity loss.
    """
    B, C, H, W = features.shape

    # Reduce spatial dimensions via pooling to limit memory usage
    if reduce_dim and H > pool_size and W > pool_size:
        features = nn.functional.adaptive_avg_pool2d(features, (pool_size, pool_size))

    # Flatten spatial dimensions
    reshaped = features.view(B, C, -1)  # (B, C, pooled_H * pooled_W)

    base_chunk_size = C // num_chunks
    chunk_sizes = [base_chunk_size] * (num_chunks - 1) + [base_chunk_size + C % num_chunks]

    loss = 0
    start = 0
    for chunk_size in chunk_sizes:
        end = min(start + chunk_size, C)
        # Compute Gram matrix
        gram_matrix = torch.bmm(reshaped[:, start:end, :], reshaped[:, start:end, :].transpose(1, 2))  # (B, C, C)

        # Identity matrix for perfect orthogonality
        channels = end - start
        identity = torch.eye(channels).to(features.device)

        # Loss: Penalize non-diagonal elements
        max_loss = 1
        if norm:
            # max_gram = torch.ones((B, channels, channels), device=features.device)
            # max_loss = torch.mean((max_gram - identity) ** 2)
            max_loss = (channels ** 2 - channels) if channels > 1 else 1
        chunk_loss = torch.mean((gram_matrix - identity) ** 2) / max_loss
        assert not norm or 0 <= chunk_loss <= 1, chunk_loss
        loss += chunk_loss
        start += chunk_size

    return loss / num_chunks


def contrastive_feature_loss(features):
    B, C, H, W = features.shape
    reshaped = features.view(B, C, -1)
    normalized = F.normalize(reshaped, p=2, dim=2)  # L2 normalization
    cosine_sim = torch.bmm(normalized, normalized.transpose(1, 2))  # Cosine similarity
    loss = torch.mean(cosine_sim)  # Encourage diversity by reducing similarity
    return loss


def determinant_loss(features, epsilon=1e-5):
    B, C, H, W = features.shape
    reshaped = features.view(B, C, -1)
    covariance = torch.bmm(reshaped, reshaped.transpose(1, 2))  # Covariance matrix
    identity = torch.eye(C).to(features.device)
    det_term = torch.det(covariance + epsilon * identity)  # Compute determinant
    loss = -torch.mean(torch.log(det_term + epsilon))  # Maximize determinant
    return loss


DIVERSITY_FUNCS = {
    "ortho": orthogonality_loss,
    "contrast": contrastive_feature_loss,
    "determ": determinant_loss
}


class FeatureDiversityLoss:
    """ Computes and accumulates diversity loss for intermediate feature maps using hooks. """

    def __init__(self, loss_fn=orthogonality_loss, alpha=0.1, to_hook=(nn.Conv2d, nn.BatchNorm2d)):
        self.loss_fn = loss_fn
        self.alpha = alpha  # Weighting factor
        self.loss = 0  # Accumulated loss
        self.hooks = []  # Stores hook handles
        self.to_hook = to_hook
        self.samples = 0

    def hook_layer(self, layer):
        return isinstance(layer, self.to_hook)

    def register_hooks(self, model):
        """
        Registers hooks to all convolutional, linear, and batch normalization layers automatically.
        """

        def hook_fn(module, input, output):
            """ Computes diversity loss for each intermediate feature map and accumulates it. """
            if not module.training:  # Skip updating during eval mode
                return
            if isinstance(output, torch.Tensor):  # Ensure output is a tensor
                self.samples += 1
                self.loss += (self.alpha * self.loss_fn(output.detach()) - self.loss) / self.samples

        # Register hooks to all relevant layers
        for layer in model.modules():
            if self.hook_layer(layer):
                self.hooks.append(layer.register_forward_hook(hook_fn))

    def remove_hooks(self):
        """ Removes hooks after training to avoid memory issues. """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset_loss(self):
        """ Resets the accumulated loss before each forward pass. """
        self.loss = 0
        self.samples = 0

    def get_loss(self):
        """ Returns the accumulated loss. """
        return self.loss


class DummyFDL:
    def __init__(self, loss_fn=lambda x: x, alpha=0.1, to_hook=()):
        pass
    def get_loss(self):
        return torch.tensor(0)
    def hook_layer(self, layer):
        return False
    def reset_loss(self):
        return
    def register_hooks(self, model):
        return
    def remove_hooks(self):
        return
