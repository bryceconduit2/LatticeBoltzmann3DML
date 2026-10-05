import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. Point Cloud Geometry Encoder (PointNet Architecture)
# -----------------------------------------------------------------------------


class PointCloudEncoder(nn.Module):
  """PointNet-style encoder that maps unstructured point clouds [B, N, 2] to a physics-aware latent vector z_geo."""

  def __init__(self, in_dim: int = 2, latent_dim: int = 16):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.GELU(),
        nn.Linear(64, 128),
        nn.GELU(),
        nn.Linear(128, 256),
        nn.GELU(),
        nn.Linear(256, latent_dim),
    )

  def forward(self, points: torch.Tensor) -> torch.Tensor:
    # points shape: [B, N_points, in_dim]
    x = self.mlp(points)  # [B, N_points, latent_dim]
    z_geo, _ = torch.max(
        x, dim=1
    )  # Permutation-invariant Global Max Pooling -> [B, latent_dim]
    return z_geo


# -----------------------------------------------------------------------------
# 2. Joint LPM Model (Point Cloud + Physics Parameter -> Flow Field)
# -----------------------------------------------------------------------------


class JointPointCloudLPM(nn.Module):

  def __init__(
      self, point_dim: int = 2, geo_dim: int = 16, phys_dim: int = 8
  ):
    super().__init__()
    self.geo_encoder = PointCloudEncoder(in_dim=point_dim, latent_dim=geo_dim)
    self.bc_encoder = nn.Sequential(
        nn.Linear(1, 16), nn.GELU(), nn.Linear(16, phys_dim)
    )

    total_latent = geo_dim + phys_dim
    self.trunk_fc = nn.Linear(total_latent, 64 * 8 * 8)
    self.flow_decoder = nn.Sequential(
        nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        ),  # [B, 32, 16, 16]
        nn.GELU(),
        nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1
        ),  # [B, 16, 32, 32]
        nn.GELU(),
        nn.ConvTranspose2d(
            16, 3, kernel_size=4, stride=2, padding=1
        ),  # [B, 3, 64, 64] -> (u_x, u_y, p)
    )

  def forward(
      self, point_cloud: torch.Tensor, reynolds: torch.Tensor
  ) -> torch.Tensor:
    z_geo = self.geo_encoder(point_cloud)
    z_phys = self.bc_encoder(reynolds)
    z = torch.cat([z_geo, z_phys], dim=-1)

    x = F.gelu(self.trunk_fc(z)).view(-1, 64, 8, 8)
    return self.flow_decoder(x)


# -----------------------------------------------------------------------------
# 3. Dataset Generator with Unstructured Point Cloud Sampling
# -----------------------------------------------------------------------------


def generate_pointcloud_flow_dataset(
    num_samples: int = 64, num_points: int = 256, res: int = 64
):
  x = torch.linspace(-1, 1, res)
  y = torch.linspace(-1, 1, res)
  xx, yy = torch.meshgrid(x, y, indexing='ij')

  point_clouds, re_list, flows = [], [], []
  for _ in range(num_samples):
    r = np.random.uniform(0.15, 0.40)
    re = np.random.uniform(100, 1000)

    # Sample N boundary surface coordinates (x, y) for the point cloud
    theta = torch.linspace(0, 2 * math.pi, num_points)
    px = r * torch.cos(theta)
    py = r * torch.sin(theta)
    point_cloud = torch.stack([px, py], dim=-1)

    # Compute ground truth physical flow field (u_x, u_y, p)
    dist = torch.clamp(torch.sqrt(xx**2 + yy**2) - r, min=0.0)
    u_x = (1.0 - torch.exp(-dist * (re / 250.0))) * (dist > 0).float()
    u_y = (
        torch.sin(math.pi * xx)
        * torch.exp(-dist * 3.0)
        * (1.0 - (re / 1200.0))
        * (dist > 0).float()
    )
    p = torch.exp(-dist * 4.0) * (dist > 0).float()

    point_clouds.append(point_cloud)
    re_list.append(torch.tensor([re / 1000.0], dtype=torch.float32))
    flows.append(torch.stack([u_x, u_y, p], dim=0))

  return (
      torch.stack(point_clouds),
      torch.stack(re_list),
      torch.stack(flows),
  )


# -----------------------------------------------------------------------------
# 4. Training Loop & Visual Validation
# -----------------------------------------------------------------------------

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  res = 64
  epochs = 250
  num_points = 256

  pcs, res_num, targets = generate_pointcloud_flow_dataset(
      num_samples=64, num_points=num_points, res=res
  )
  pcs, res_num, targets = (
      pcs.to(device),
      res_num.to(device),
      targets.to(device),
  )

  model = JointPointCloudLPM(point_dim=2, geo_dim=16, phys_dim=8).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

  print('Training PointCloud-conditioned Physics Model...')
  for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    predictions = model(pcs, res_num)

    # CRITICAL: Loss is calculated ONLY on downstream flow field prediction.
    # Gradients flow backwards directly into the PointNet PointCloudEncoder.
    loss = F.mse_loss(predictions, targets)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == 1:
      print(f'Epoch {epoch:03d}/{epochs} | Flow MSE Loss: {loss.item():.6f}')

  # Evaluation & Visualization
  test_idx = 0
  pc_val = pcs[test_idx : test_idx + 1]
  re_val = res_num[test_idx : test_idx + 1]
  target_val = targets[test_idx].cpu().detach().numpy()

  pred_val = model(pc_val, re_val)[0].cpu().detach().numpy()
  pc_np = pc_val[0].cpu().detach().numpy()

  gt_ux = target_val[0]
  pred_ux = pred_val[0]
  abs_err = np.abs(gt_ux - pred_ux)

  fig, axes = plt.subplots(1, 4, figsize=(18, 4))

  # Panel 1: Input Point Cloud
  axes[0].scatter(pc_np[:, 0], pc_np[:, 1], c='red', s=12)
  axes[0].set_xlim([-1, 1])
  axes[0].set_ylim([-1, 1])
  axes[0].set_aspect('equal')
  axes[0].set_title(f'Input Point Cloud ($N={num_points}$)')
  axes[0].set_xlabel('X')
  axes[0].set_ylabel('Y')

  # Panel 2: CFD Target Field
  im1 = axes[1].imshow(
      gt_ux, cmap='viridis', origin='lower', extent=[-1, 1, -1, 1]
  )
  axes[1].set_title('Target Flow Field $u_x$')
  axes[1].set_xlabel('X')
  axes[1].set_ylabel('Y')
  plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

  # Panel 3: LPM Prediction driven by Point Cloud Latent
  im2 = axes[2].imshow(
      pred_ux, cmap='viridis', origin='lower', extent=[-1, 1, -1, 1]
  )
  axes[2].set_title('LPM Prediction $\hat{u}_x$')
  axes[2].set_xlabel('X')
  axes[2].set_ylabel('Y')
  plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

  # Panel 4: Absolute Error
  im3 = axes[3].imshow(
      abs_err, cmap='inferno', origin='lower', extent=[-1, 1, -1, 1]
  )
  axes[3].set_title('Absolute Error $|u_x - \hat{u}_x|$')
  axes[3].set_xlabel('X')
  axes[3].set_ylabel('Y')
  plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

  plt.tight_layout()
  plt.show()