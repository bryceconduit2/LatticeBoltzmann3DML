import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        real_weights = torch.randn(in_channels, out_channels, modes, modes, modes) * scale
        imag_weights = torch.randn(in_channels, out_channels, modes, modes, modes) * scale
        self.weights = nn.Parameter(torch.complex(real_weights, imag_weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.size(0)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            batchsize, self.weights.size(1), x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes, :self.modes, :self.modes] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :self.modes, :self.modes, :self.modes],
            self.weights
        )
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), dim=[-3, -2, -1])

class PINO3d(nn.Module):
    def __init__(self, modes: int = 8, width: int = 32):
        super().__init__()
        self.fc0 = nn.Linear(4, width)
        self.conv0 = SpectralConv3d(width, width, modes)
        self.w0 = nn.Conv3d(width, width, 1)
        self.conv1 = SpectralConv3d(width, width, modes)
        self.w1 = nn.Conv3d(width, width, 1)
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x).permute(0, 4, 1, 2, 3)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = x.permute(0, 2, 3, 4, 1)
        return self.fc2(F.gelu(self.fc1(x)))

def compute_3d_spectral_laplacian(u: torch.Tensor, L: float = 1.0) -> torch.Tensor:
    batch, X, Y, Z, _ = u.shape
    u_grid = u.squeeze(-1)

    kx = torch.fft.fftfreq(X, d=L / X, device=u.device) * 2 * math.pi
    ky = torch.fft.fftfreq(Y, d=L / Y, device=u.device) * 2 * math.pi
    kz = torch.fft.rfftfreq(Z, d=L / Z, device=u.device) * 2 * math.pi

    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = KX**2 + KY**2 + KZ**2

    u_ft = torch.fft.rfftn(u_grid, dim=[-3, -2, -1])
    laplacian_ft = -k_squared * u_ft
    return torch.fft.irfftn(laplacian_ft, s=(X, Y, Z), dim=[-3, -2, -1]).unsqueeze(-1)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = 64
    epochs = 300

    grid_1d = torch.linspace(0, 1, res + 1, device=device)[:-1]
    gx, gy, gz = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)

    f_field = (torch.sin(2 * math.pi * gx) * torch.sin(2 * math.pi * gy) * torch.sin(2 * math.pi * gz)).unsqueeze(-1).unsqueeze(0)
    u_ground_truth = f_field / (3 * (2 * math.pi)**2)

    pino_input = torch.cat([grid, f_field], dim=-1)
    model = PINO3d(modes=8, width=32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        u_pred = model(pino_input)

        data_loss = torch.mean((u_pred - u_ground_truth)**2) / torch.mean(u_ground_truth**2)
        laplacian_u = compute_3d_spectral_laplacian(u_pred)
        pde_residual = -laplacian_u - f_field
        pde_loss = torch.mean(pde_residual**2)

        loss = data_loss + 1e-4 * pde_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    # SLICE AT z = 0.25 (index 16) WHERE WAVE AMPLITUDE IS AT MAXIMUM
    z_slice = res // 4  
    
    gt_slice = u_ground_truth[0, :, :, z_slice, 0].cpu().detach().numpy()
    pred_slice = u_pred[0, :, :, z_slice, 0].cpu().detach().numpy()
    error_slice = np.abs(gt_slice - pred_slice)
    pde_res_slice = np.abs(pde_residual[0, :, :, z_slice, 0].cpu().detach().numpy())

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    titles = [f"Ground Truth $u(x,y,z_{{0.25}})$", f"3D PINO Prediction $\hat{{u}}$",
              "Absolute Error $|u - \hat{u}|$", "PDE Residual $|-\\nabla^2 \\hat{u} - f|$"]
    data_slices = [gt_slice, pred_slice, error_slice, pde_res_slice]
    cmaps = ['viridis', 'viridis', 'magma', 'inferno']

    for ax, slice_data, title, cmap in zip(axes, data_slices, titles, cmaps):
        im = ax.imshow(slice_data, cmap=cmap, origin='lower')
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()