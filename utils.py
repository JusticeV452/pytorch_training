import json
import random
import torch
import torch.nn.functional as F

from torch import nn


def load_json(path):
    with open(path, 'r', encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    with open(path, 'w+', encoding="utf-8") as file:
        json.dump(data, file)


def np_nd_apply(arr, transform_func, *args):
    parts = []
    for i in range(0, len(arr), 4):
        parts.append(transform_func(arr[i:i + 3], *args))
        parts.append(arr[i + 3].unsqueeze(0))
    return torch.concat(parts, axis=0)


def percent_chance(chance):
    return random.uniform(0, 1) <= chance

def permute(items):
    random.shuffle(items)
    return items

def int_round(n, places=None):
    return int(round(n, places))


def rec_int_pow(num, base, power):
    if power < 0:
        power = -power
        base = base ** -1
    result = num
    for _ in range(power):
        result = int_round(result * base)
    return result


def next_largest_dividend(n, div):
    if n % div == 0:
        return n
    return n + (div - n % div)


def next_smallest_dividend(n, div):
    if n % div == 0:
        return n
    return n - n % div


def get_group_norm(num_groups):
    return lambda outc: nn.GroupNorm(num_groups, outc)


def spatial_pad_to(t, shape):
    diffY = shape[2] - t.size()[2]
    diffX = shape[3] - t.size()[3]

    return F.pad(t, [
        diffX // 2, diffX - diffX // 2,
        diffY // 2, diffY - diffY // 2
    ])


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Conv1d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def vae_linear_init(m):
    weights_init(m)
    classname = m.__class__.__name__
    if "Linear" in classname:
        nn.init.normal_(m.weight.data, 0, 0.001)


def parse_filter_size(filter_size):
    return (filter_size, filter_size) if isinstance(filter_size, int) else filter_size


def safe_dict_pop(D, key, default=None):
    if key in D:
        return D.pop(key)
    return default


def channel_width_flatten(tensor):
    """
    Reshapes a tensor of images concatenated in the channel dimension 
    into a tensor with images concatenated horizontally.

    Args:
        tensor (torch.Tensor): Input tensor of shape (N, C, H, W), 
                               where C = num_images * num_channels_per_image.

    Returns:
        torch.Tensor: Reshaped tensor with images concatenated horizontally.
    """
    C = tensor.shape[1]
    
    # Determine number of images (assuming RGB, so C should be divisible by 3)
    num_channels_per_image = 3  # For RGB images
    num_images = C // num_channels_per_image

    if C % num_channels_per_image != 0:
        raise ValueError(f"Invalid number of channels: {C}. Must be divisible by {num_channels_per_image}.")

    # Split tensor into `num_images` along channel dimension
    images = torch.chunk(tensor, chunks=num_images, dim=1)  # List of (N, 3, H, W) tensors

    # Concatenate along width (dim=3)
    reshaped_tensor = torch.cat(images, dim=3)  # Shape: (N, 3, H, W * num_images)

    return reshaped_tensor


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_model_size_bytes(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return param_size + buffer_size


def calc_model_size(model):
    size_all_mb = calc_model_size_bytes(model) / 1024**2
    return '{:.3f}MB'.format(size_all_mb)


def shuffle_tensor(t, idx=None, return_idx=False):
    if not return_idx:
        return t
    return t, idx
    # if type(idx) is type(None):
    #     idx = torch.randperm(t.shape[0])
    # shuffled = t[idx]
    # if not return_idx:
    #     return shuffled
    # return shuffled, idx
