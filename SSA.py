import torch
import torch.nn as nn
import torch.nn.functional as F


class SSA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.diff_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _smoothmax_pool_h(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.softmax(F.softplus(self.beta) * x, dim=2)
        return (alpha * x).sum(dim=2, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M_avg = self.avg_pool(x)
        M_smoothmax = self._smoothmax_pool_h(x)
        M_diff = M_smoothmax - M_avg
        G = torch.sigmoid(
            self.conv(M_avg + M_smoothmax) +
            self.conv(M_diff) * self.diff_scale
        )
        return x * G
