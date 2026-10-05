import torch
import torch.nn.functional as F
from torch import nn


def _boundary_layer_mask(obstacle, adjacency, width):
    """Return fluid nodes within ``width`` graph cells of the obstacle."""
    expanded = obstacle.clone()
    for _ in range(max(0, width)):
        expanded = expanded | expanded[adjacency[1:]].any(dim=0)
    return expanded & ~obstacle


def _interpolate_lbm_field(field, target_shape):
    """Interpolate a flattened LBM field onto a target z-y-x lattice."""
    if field.ndim == 1:
        field = field.unsqueeze(0)
    channels = field.shape[0]
    reshaped = field.reshape(1, channels, *field.shape[1:])
    interpolated = F.interpolate(
        reshaped, size=target_shape, mode="trilinear", align_corners=True
    )
    return interpolated.reshape(channels, -1)


def _restrict_lbm_field(field, source_shape, target_shape):
    """Average a flattened fine-grid field onto a coarse z-y-x lattice."""
    if field.ndim == 1:
        field = field.unsqueeze(0)
    channels = field.shape[0]
    reshaped = field.reshape(1, channels, *source_shape)
    restricted = F.adaptive_avg_pool3d(reshaped, target_shape)
    return restricted.reshape(channels, -1)


def build_geometry_representations(obstacle, adjacency, positions, max_distance=16):
    """Build geometry channels that can be selected by the learned encoder."""
    dtype = positions.dtype
    obstacle = obstacle.bool()
    occupancy = obstacle.to(dtype)
    boundary = (~obstacle) & obstacle[adjacency[1:]].any(dim=0)
    distance = torch.full(
        (obstacle.numel(),), float(max_distance), device=obstacle.device,
        dtype=dtype,
    )
    distance[obstacle] = 0.0
    reached = obstacle.clone()
    for hop in range(1, max_distance + 1):
        next_reached = reached | reached[adjacency[1:]].any(dim=0)
        newly_reached = next_reached & ~reached
        distance[newly_reached] = float(hop)
        reached = next_reached
    normalized_distance = (distance / max(float(max_distance), 1.0)).clamp(0.0, 1.0)
    return torch.stack([occupancy, boundary.to(dtype), normalized_distance], dim=1)


class FlowConditionedGeometryEncoder(nn.Module):
    """Learn which geometry channels are useful for the local flow state."""

    def __init__(self, num_representations=3, latent_dim=8, hidden_dim=16):
        super().__init__()
        self.num_representations = num_representations
        self.latent_dim = latent_dim
        self.gate = nn.Sequential(
            nn.Linear(num_representations + 4, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_representations),
        )
        self.project = nn.Linear(num_representations, latent_dim)

    def forward(self, representations, flow_features):
        if representations.ndim != 2 or flow_features.ndim != 2:
            raise ValueError("geometry and flow features must be rank-2 tensors")
        if representations.shape[1] != self.num_representations:
            raise ValueError("unexpected number of geometry representations")
        if flow_features.shape[1] != 4:
            raise ValueError("flow_features must contain rho and 3 velocity channels")
        gate_input = torch.cat([representations, flow_features], dim=1)
        weights = torch.softmax(self.gate(gate_input), dim=1)
        return self.project(representations * weights), weights


class LBMGraphSuperResolutionGNN(nn.Module):
    """Refine a coarse LBM field onto a fine graph."""

    def __init__(self, hidden_dim=32, max_correction=0.05, boundary_layer_width=2,
                 boundary_layer_boost=2.0, geometry_latent_dim=0,
                 geometry_representation_count=3):
        super().__init__()
        self.max_correction = max_correction
        self.boundary_layer_width = boundary_layer_width
        self.boundary_layer_boost = boundary_layer_boost
        self.geometry_latent_dim = geometry_latent_dim
        self.geometry_encoder = (
            FlowConditionedGeometryEncoder(
                num_representations=geometry_representation_count,
                latent_dim=geometry_latent_dim,
            )
            if geometry_latent_dim > 0 else None
        )
        encoder_input_dim = 9 + geometry_latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, coarse_rho, coarse_velocity, obstacle, inlet, adjacency,
                positions, geometry_features=None, return_geometry_weights=False):
        position_scale = positions.amax(dim=0).clamp_min(1.0)
        normalized_positions = positions / position_scale
        features = torch.cat([
            coarse_rho.unsqueeze(1), coarse_velocity.transpose(0, 1),
            obstacle.to(coarse_rho.dtype).unsqueeze(1),
            inlet.to(coarse_rho.dtype).unsqueeze(1), normalized_positions,
        ], dim=1)
        geometry_weights = None
        if self.geometry_encoder is not None:
            if geometry_features is None:
                raise ValueError("geometry_features are required when geometry learning is enabled")
            geometry_latent, geometry_weights = self.geometry_encoder(
                geometry_features.to(features),
                torch.cat([coarse_rho.unsqueeze(1), coarse_velocity.transpose(0, 1)], dim=1),
            )
            features = torch.cat([features, geometry_latent], dim=1)
        encoded = self.encoder(features)
        neighbors = encoded[adjacency[1:]]
        center = encoded.unsqueeze(0).expand(neighbors.shape[0], -1, -1)
        messages = self.message(torch.cat([center, neighbors], dim=-1)).mean(dim=0)
        correction = self.max_correction * torch.tanh(
            self.decoder(torch.cat([encoded, messages], dim=-1))
        )
        boundary_layer = _boundary_layer_mask(
            obstacle, adjacency, self.boundary_layer_width
        )
        correction_scale = torch.where(
            boundary_layer,
            torch.full_like(boundary_layer, self.boundary_layer_boost, dtype=coarse_rho.dtype),
            torch.ones_like(coarse_rho),
        )
        correction = correction * correction_scale.unsqueeze(1)
        correction[obstacle] = 0.0
        correction[inlet] = 0.0
        result = (coarse_velocity + correction.transpose(0, 1)).masked_fill(
            obstacle.unsqueeze(0), 0.0
        )
        if return_geometry_weights:
            return result, geometry_weights
        return result


def predict_fine_lbm_velocity(coarse_solver, fine_solver, superres_model, positions,
                              coarse_shape, fine_shape, geometry_features=None):
    """Predict a fine-grid velocity field from a coarse solver state."""
    coarse_rho = _interpolate_lbm_field(
        coarse_solver.rho.reshape(1, *coarse_shape), fine_shape
    )[0]
    coarse_velocity = _interpolate_lbm_field(
        coarse_solver.u.reshape(3, *coarse_shape), fine_shape
    )
    return superres_model(
        coarse_rho, coarse_velocity, fine_solver.obstacle, fine_solver.inlet,
        fine_solver.adjacency, positions, geometry_features=geometry_features,
    )


def predict_fine_lbm_density(coarse_solver, coarse_shape, fine_shape):
    """Interpolate coarse density onto the fine lattice for pressure estimates."""
    return _interpolate_lbm_field(
        coarse_solver.rho.reshape(1, *coarse_shape), fine_shape
    )[0]


def restrict_fine_lbm_velocity(fine_velocity, fine_shape, coarse_shape):
    """Restrict a fine velocity field to the coarse lattice for feedback."""
    return _restrict_lbm_field(fine_velocity, fine_shape, coarse_shape)


def restrict_fine_lbm_velocity_residual(
    fine_velocity, coarse_velocity, fine_shape, coarse_shape, max_delta=None
):
    """Restrict only the learned fine-grid correction to the coarse lattice."""
    coarse_velocity_fine = _interpolate_lbm_field(
        coarse_velocity.reshape(3, *coarse_shape), fine_shape
    )
    fine_residual = fine_velocity - coarse_velocity_fine
    coarse_residual = _restrict_lbm_field(
        fine_residual, fine_shape, coarse_shape
    )
    if max_delta is not None:
        residual_norm = torch.linalg.vector_norm(coarse_residual, dim=0)
        scale = (max_delta / residual_norm.clamp_min(1e-12)).clamp(max=1.0)
        coarse_residual = coarse_residual * scale.unsqueeze(0)
    return coarse_velocity + coarse_residual


def train_lbm_superresolution_gnn(
    coarse_solver, fine_solver, superres_model, steps=200, epochs=50,
    learning_rate=1e-3, positions=None, coarse_shape=None, fine_shape=None,
    geometry_features=None,
):
    """Train coarse-to-fine velocity reconstruction from paired LBM states."""
    if positions is None or coarse_shape is None or fine_shape is None:
        raise ValueError("positions, coarse_shape, and fine_shape are required")
    superres_model.train()
    with torch.no_grad():
        for _ in range(steps):
            coarse_solver.step()
            fine_solver.step()

    optimizer = torch.optim.Adam(superres_model.parameters(), lr=learning_rate)
    target = fine_solver.u.detach()
    fluid = ~fine_solver.obstacle
    boundary_layer = _boundary_layer_mask(
        fine_solver.obstacle, fine_solver.adjacency,
        superres_model.boundary_layer_width,
    )
    loss_weight = torch.where(
        boundary_layer,
        torch.full_like(target[0], superres_model.boundary_layer_boost),
        torch.ones_like(target[0]),
    )
    for _ in range(epochs):
        prediction = predict_fine_lbm_velocity(
            coarse_solver, fine_solver, superres_model, positions,
            coarse_shape, fine_shape, geometry_features,
        )
        squared_error = (prediction - target) ** 2
        loss = (squared_error * loss_weight.unsqueeze(0))[..., fluid].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(superres_model.parameters(), 1.0)
        optimizer.step()
    superres_model.eval()
    return float(loss.detach().cpu())


def save_lbm_superresolution_gnn(model, path):
    hidden_dim = model.encoder[0].out_features
    torch.save({
        "state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "max_correction": model.max_correction,
        "boundary_layer_width": model.boundary_layer_width,
        "boundary_layer_boost": model.boundary_layer_boost,
        "geometry_latent_dim": model.geometry_latent_dim,
        "geometry_representation_count": (
            model.geometry_encoder.num_representations
            if model.geometry_encoder is not None else 3
        ),
    }, path)


def load_lbm_superresolution_gnn(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = LBMGraphSuperResolutionGNN(
        hidden_dim=checkpoint["hidden_dim"],
        max_correction=checkpoint["max_correction"],
        boundary_layer_width=checkpoint.get("boundary_layer_width", 2),
        boundary_layer_boost=checkpoint.get("boundary_layer_boost", 2.0),
        geometry_latent_dim=checkpoint.get("geometry_latent_dim", 0),
        geometry_representation_count=checkpoint.get("geometry_representation_count", 3),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


