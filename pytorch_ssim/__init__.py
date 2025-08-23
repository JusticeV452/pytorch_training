import torch
import torch.nn.functional as F

from math import exp
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel, dtype=torch.float32):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).to(dtype).unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, reduce_mode = "mean"):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if reduce_mode == "mean":
        return ssim_map.mean()
    elif reduce_mode == "map":
        return ssim_map
    else:
        return ssim_map.mean(1).mean(1).mean(1).sum()


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True, dtype=torch.float32, reduce_mode=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average if reduce_mode is None else (reduce_mode == "mean")
        self.reduce_mode = ("mean" if size_average else "sum") if reduce_mode is None else reduce_mode
        self.channel = 1
        self.dtype = dtype
        self.window = create_window(window_size, self.channel, self.dtype)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.dtype)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.reduce_mode)


def ssim(img1, img2, window_size = 11, size_average = True, dtype=torch.float32, reduce_mode=None):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel, dtype)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    if reduce_mode is None:
        reduce_mode = "mean" if size_average else "sum"

    return _ssim(img1, img2, window, window_size, channel, reduce_mode)


class NSSIM(SSIM):
    def __init__(self, window_size = 11, size_average = True, dtype=torch.float32, reduce_mode=None):
        super().__init__(window_size, size_average, dtype, reduce_mode)
    def forward(self, img1, img2, *imgs,):
        total_ssim = 0
        all_imgs = [img1, img2, *imgs]
        for i, img in enumerate(all_imgs):
            total_ssim += super().forward(all_imgs[i - 1], img)
        return total_ssim / len(all_imgs)
