import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# 1. Modular Geometry & Flow Dataset Generators
# -----------------------------------------------------------------------------

def sdf_circle(xx, yy, cx=0.0, cy=0.0, r=0.25):
    return torch.sqrt((xx - cx)**2 + (yy - cy)**2) - r

def sdf_ellipse(xx, yy, rx=0.35, ry=0.18):
    return torch.sqrt((xx / rx)**2 + (yy / ry)**2) - 1.0

def sdf_dual_circles(xx, yy, r=0.18, offset=0.35):
    d1 = torch.sqrt((xx - offset)**2 + (yy - offset)**2) - r
    d2 = torch.sqrt((xx + offset)**2 + (yy + offset)**2) - r
    return torch.minimum(d1, d2)

def sdf_box(xx, yy, size_x=0.25, size_y=0.25):
    dx = torch.abs(xx) - size_x
    dy = torch.abs(yy) - size_y
    return torch.maximum(dx, dy)

def generate_flow_from_sdf(sdf, re_scaled, res=64):
    """Generates synthetic physics flow field vectors (u_x, u_y, p) from an SDF."""
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, res), torch.linspace(-1, 1, res), indexing='ij')
    dist = torch.clamp(sdf, min=0.0)
    
    # Physics wake approximation dependent on geometry and Reynolds number
    u_x = (1.0 - torch.exp(-dist * (re_scaled * 4.0))) * (sdf > 0).float()
    u_y = torch.sin(math.pi * xx) * torch.exp(-dist * 2.5) * (sdf > 0).float()
    p = torch.exp(-dist * 3.5) * (sdf > 0).float()
    
    return torch.stack([u_x, u_y, p], dim=0)

# -----------------------------------------------------------------------------
# 2. Joint LPM Architecture & Training
# -----------------------------------------------------------------------------

class GeometryEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.fc = nn.Linear(32 * 8 * 8, latent_dim)

    def forward(self, sdf):
        return self.fc(torch.flatten(self.conv(sdf), start_dim=1))

class JointLPM(nn.Module):
    def __init__(self, geo_dim=16, phys_dim=8):
        super().__init__()
        self.geo_enc = GeometryEncoder(geo_dim)
        self.bc_enc = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, phys_dim))
        self.trunk = nn.Linear(geo_dim + phys_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)
        )

    def forward(self, sdf, reynolds):
        z = torch.cat([self.geo_enc(sdf), self.bc_enc(reynolds)], dim=-1)
        x = F.gelu(self.trunk(z)).view(-1, 64, 8, 8)
        return self.decoder(x)

# -----------------------------------------------------------------------------
# 3. Execution: Training on Circles, Testing on Unseen Shapes
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = 64
    x = torch.linspace(-1, 1, res)
    xx, yy = torch.meshgrid(x, x, indexing='ij')

    # A. Generate Training Set (ONLY Single Circles, Re = 100..500)
    train_sdfs, train_res, train_flows = [], [], []
    for _ in range(128):
        r = np.random.uniform(0.15, 0.35)
        re = np.random.uniform(0.1, 0.5) # Scaled Re
        sdf = sdf_circle(xx, yy, r=r)
        
        train_sdfs.append(sdf.unsqueeze(0))
        train_res.append(torch.tensor([re], dtype=torch.float32))
        train_flows.append(generate_flow_from_sdf(sdf, re, res))

    train_sdfs = torch.stack(train_sdfs).to(device)
    train_res = torch.stack(train_res).to(device)
    train_flows = torch.stack(train_flows).to(device)

    # Train Model
    model = JointLPM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    
    print("Training on single circle geometries...")
    for epoch in range(1, 201):
        optimizer.zero_grad()
        loss = F.mse_loss(model(train_sdfs, train_res), train_flows)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Train MSE Loss: {loss.item():.6f}")

    # B. Generate Generalization Test Suite (Unseen Topologies & Extrapolated Re)
    test_cases = [
        ("In-Dist Circle (Re=0.3)", sdf_circle(xx, yy, r=0.25), 0.3),
        ("Unseen Ellipse (Re=0.4)", sdf_ellipse(xx, yy, rx=0.35, ry=0.18), 0.4),
        ("Dual Cylinders (Re=0.5)", sdf_dual_circles(xx, yy, r=0.15, offset=0.3), 0.5),
        ("Unseen Square (Re=0.8 - Extrapolated)", sdf_box(xx, yy, size_x=0.2, size_y=0.2), 0.8)
    ]

    # C. Evaluate & Plot Generalization
    fig, axes = plt.subplots(4, 3, figsize=(12, 12))

    for idx, (name, test_sdf, test_re) in enumerate(test_cases):
        test_sdf_t = test_sdf.unsqueeze(0).unsqueeze(0).to(device)
        test_re_t = torch.tensor([[test_re]], dtype=torch.float32).to(device)
        
        gt_flow = generate_flow_from_sdf(test_sdf, test_re, res)
        with torch.no_grad():
            pred_flow = model(test_sdf_t, test_re_t)[0].cpu()

        gt_ux = gt_flow[0].numpy()
        pred_ux = pred_flow[0].numpy()
        rel_err = np.linalg.norm(gt_ux - pred_ux) / np.linalg.norm(gt_ux)

        # Plot Ground Truth, Prediction, and Error Map
        im0 = axes[idx, 0].imshow(gt_ux, cmap='viridis', origin='lower')
        axes[idx, 0].set_title(f"{name}\nGT $u_x$")
        plt.colorbar(im0, ax=axes[idx, 0])

        im1 = axes[idx, 1].imshow(pred_ux, cmap='viridis', origin='lower')
        axes[idx, 1].set_title(f"Predicted $u_x$\n(Rel $L_2$ Error: {rel_err:.2%})")
        plt.colorbar(im1, ax=axes[idx, 1])

        im2 = axes[idx, 2].imshow(np.abs(gt_ux - pred_ux), cmap='inferno', origin='lower')
        axes[idx, 2].set_title("Abs Error $|u_x - \hat{u}_x|$")
        plt.colorbar(im2, ax=axes[idx, 2])

    plt.tight_layout()
    plt.show()