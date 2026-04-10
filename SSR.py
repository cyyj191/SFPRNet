ssr
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class SSR(nn.Module):
    def __init__(self, F_g, F_l, F_int=None, kernel_size=3, groups='auto',
                 psi_kernel_h=11, psi_kernel_w=7):
        super().__init__()

        if F_int is None:
            F_int = max(8, F_g // 2)

        if groups == 'auto':
            if F_int <= 32:
                groups = 1
            else:
                groups = max(1, F_int // 16)

        F_int = (F_int // groups) * groups
        padding = kernel_size // 2

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, padding=padding, groups=groups, bias=True),
            LayerNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, padding=padding, groups=groups, bias=True),
            LayerNorm2d(F_int)
        )

        self.act = nn.GELU()

        kh = int(psi_kernel_h)
        if kh % 2 == 0:
            kh += 1
        kw = int(psi_kernel_w)
        if kw % 2 == 0:
            kw += 1

        self.psi_v = nn.Conv2d(
            F_int, F_int,
            kernel_size=(kh, 1),
            padding=(kh // 2, 0),
            groups=F_int,
            bias=True
        )
        self.psi_h = nn.Conv2d(
            F_int, F_int,
            kernel_size=(1, kw),
            padding=(0, kw // 2),
            groups=F_int,
            bias=True
        )

        self.psi_fusion_norm = LayerNorm2d(F_int)
        self.psi_fusion_conv = nn.Conv2d(F_int, 1, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        with torch.no_grad():
            self.psi_fusion_conv.weight.mul_(1e-3)
            self.psi_fusion_conv.bias.fill_(2.0)

    def forward(self, g, x):
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        u = self.act(g1 + x1)

        joint = self.psi_v(u) + self.psi_h(u)
        joint = self.psi_fusion_norm(joint)
        mask = torch.sigmoid(self.psi_fusion_conv(joint))
        return x * mask
