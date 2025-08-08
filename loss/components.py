import torch
import torch.nn.functional as F

from torch import nn


class SOBEL(nn.Module):
    def __init__(self, device=None, dtype=torch.float32):
        super(SOBEL, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ], dtype=dtype)
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(self.device, self.dtype)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(self.device, self.dtype)

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N*C, 1, H, W), gt.reshape(N*C, 1, H, W)], 0)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N*C], sobel_stack_x[N*C:]
        pred_Y, gt_Y = sobel_stack_y[:N*C], sobel_stack_y[N*C:]

        L1X, L1Y = torch.abs(pred_X-gt_X), torch.abs(pred_Y-gt_Y)
        loss = (L1X+L1Y)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.bce_loss = nn.BCELoss(reduction='none')  # Compute per-sample BCE loss

    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        prob = predictions * targets + (1 - predictions) * (1 - targets)  # Get probability
        focal_weight = (1 - prob) ** self.gamma  # Compute focal weight
        return (focal_weight * bce).mean()  # Weighted loss
    

class TrainableLoss(nn.Module):
    def __init__(self, loss_funcs, reduce="sum", softmax=True, mag_range=1000):
        super(TrainableLoss, self).__init__()

        # Initialize trainable weights (logits) for each loss
        self.weights = nn.Parameter(torch.ones(len(loss_funcs)))
        self.loss_funcs = loss_funcs
        self.reduce = reduce
        self.softmax = softmax
        self.mag_range = mag_range

    def get_loss_weights(self):
        if self.softmax:
            # Apply softmax to logits to obtain normalized loss weights
            return torch.softmax(self.weights, dim=0)
        return [
            torch.clamp(w, 1 / self.mag_range, self.mag_range)
            for w in self.weights
        ]

    def forward(self, pred, ex):
        # Compute weighted losses
        weighted_losses = [
            w * loss_func(pred, ex)
            for w, loss_func in zip(self.get_loss_weights(), self.loss_funcs.values())
        ]
        loss_div = 1 if self.reduce == "sum" else pred.shape[0]
        loss = sum(weighted_losses) / loss_div
        if not self.softmax:
            loss -= sum(self.get_loss_weights())
        
        return loss, {
            loss_name: weighted_loss / loss_div
            for loss_name, weighted_loss in zip(self.loss_funcs, weighted_losses)
        }
