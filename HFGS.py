import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt2d(x: torch.Tensor):
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(x.shape)}")
    if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
        raise ValueError(f"Expected even spatial size, got {x.shape[-2:]}")

    x01 = x[:, :, 0::2, :] * 0.5
    x02 = x[:, :, 1::2, :] * 0.5

    a = x01[:, :, :, 0::2]
    b = x01[:, :, :, 1::2]
    c = x02[:, :, :, 0::2]
    d = x02[:, :, :, 1::2]

    ll = a + b + c + d
    hl = -a + b - c + d
    lh = -a - b + c + d
    hh = a - b - c + d
    return ll, hl, lh, hh


class HFGSDown(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hf_scale: ,
        delta_max: ,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.delta_max = float(delta_max)
        self.eps = 1e-6

        self.register_buffer(
            "range_r",
            torch.tensor([0.0, hf_scale, hf_scale, hf_scale], dtype=torch.float32).view(1, 4, 1, 1)
        )

        self.freq_theta = nn.Parameter(torch.tensor([0.0, , , ], dtype=torch.float32).view(1, 4, 1, 1))
        self.a_raw = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(-2.0))

        self.main_proj = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True),
        )

        self.aux_proj = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=True)

        self.fuse_w = nn.Parameter(torch.tensor(0.0))
        self.fuse_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_ch:
            raise RuntimeError(f"[HFGSDown] input C={x.shape[1]} != in_ch={self.in_ch}")
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            raise RuntimeError(f"[HFGSDown] expects even spatial size, got {x.shape[-2:]}")

        ll, hl, lh, hh = haar_dwt2d(x)

        high_e = (
            hl.abs().mean(dim=(1, 2, 3), keepdim=True)
            + lh.abs().mean(dim=(1, 2, 3), keepdim=True)
            + hh.abs().mean(dim=(1, 2, 3), keepdim=True)
        )
        ll_e = ll.abs().mean(dim=(1, 2, 3), keepdim=True) + self.eps
        log_s = torch.log(high_e / ll_e + self.eps)

        k = F.softplus(self.a_raw)
        gate = torch.sigmoid(k * (self.b - log_s))

        base = torch.tanh(self.freq_theta).to(dtype=x.dtype)
        r = self.range_r.to(dtype=x.dtype)

        gate_mask = torch.ones(x.shape[0], 4, 1, 1, device=x.device, dtype=x.dtype)
        gate_mask[:, 1:4] = gate

        scale = 1.0 + (r * base) * gate_mask

        sub = torch.cat([ll, hl, lh, hh], dim=1)
        h2, w2 = sub.shape[-2], sub.shape[-1]
        sub = sub.reshape(x.shape[0], 4, self.in_ch, h2, w2)
        sub = sub * scale.unsqueeze(2)
        sub = sub.reshape(x.shape[0], 4 * self.in_ch, h2, w2)

        y_main = self.main_proj(sub)
        y_aux = self.aux_proj(x)

        fuse_k = F.softplus(self.fuse_w)
        delta = self.delta_max * torch.tanh(fuse_k * log_s + self.fuse_b)
        alpha = 0.5 + delta

        return 2.0 * ((1.0 - alpha) * y_aux + alpha * y_main)


if __name__ == "__main__":
    x = torch.randn(1, 32, 256, 256)
    model = HFGSDown(32, 64)
    y = model(x)
    print(x.shape, "->", y.shape)
