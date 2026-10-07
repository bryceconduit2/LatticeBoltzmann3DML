import os
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F



def aggregate_neighbors(adjacency, h):
    """
    Aggregates node features 'h' [N, hidden_dim] across neighbors.
    Handles D3Q19 stencil index tensors [19, N] or [N, 19], 
    as well as standard [N, N] sparse/dense adjacency matrices.
    """
    N = h.size(0)
    
    # 1. Sparse matrix [N, N]
    if adjacency.is_sparse:
        return torch.sparse.mm(adjacency.float(), h)
    
    # 2. D3Q19 Stencil Index Table [19, N] or [N, 19]
    if adjacency.dim() == 2:
        r, c = adjacency.size()
        if r != N and c == N:  # Shape [19, N]
            idx = torch.clamp(adjacency.long(), 0, N - 1)
            return h[idx].sum(dim=0)  # Gathers [19, N, hidden] -> sums to [N, hidden]
        elif r == N and c != N:  # Shape [N, 19]
            idx = torch.clamp(adjacency.long(), 0, N - 1)
            return h[idx].sum(dim=1)  # Gathers [N, 19, hidden] -> sums to [N, hidden]
            
    # 3. Standard Dense Matrix [N, N]
    return torch.matmul(adjacency.float(), h)

class FlowInformedGeometryEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_embed_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.msg = nn.Linear(hidden_dim, hidden_dim)
        self.msg_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.to_latent_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, node_inputs, adjacency, obstacle_mask):
        node_inputs = node_inputs.float()
        
        # 1. Embed node features
        h = F.relu(self.node_embed_norm(self.node_embed(node_inputs)))
        
        # 2. Message passing across 19 directional neighbors
        agg = aggregate_neighbors(adjacency, h)
        h = self.msg_norm(F.relu(self.msg(agg)) + h)
        
        # 3. Global mean pooling across obstacle boundary nodes
        mask = obstacle_mask.bool().unsqueeze(-1)
        obstacle_features = torch.where(mask, h, torch.zeros_like(h))
        num_obstacle_nodes = torch.clamp(mask.sum(), min=1.0)
        
        global_shape_vector = obstacle_features.sum(dim=0, keepdim=True) / num_obstacle_nodes
        
        # 4. Latent vector z [1, latent_dim]
        latent_hidden = self.to_latent[0](global_shape_vector)
        latent_hidden = F.relu(self.to_latent_norm(latent_hidden))
        return self.to_latent[2](latent_hidden)


class FlowDecoderGNN(nn.Module):
    def __init__(self, spatial_dim=4, latent_dim=16, hidden_dim=64, out_dim=3):
        super().__init__()
        
        self.node_in = nn.Sequential(
            nn.Linear(spatial_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_in_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        self.msg1 = nn.Linear(hidden_dim, hidden_dim)
        self.msg1_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.msg2 = nn.Linear(hidden_dim, hidden_dim)
        self.msg2_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        self.node_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.node_out_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.register_buffer("output_mean", torch.zeros(out_dim))
        self.register_buffer("output_std", torch.ones(out_dim))

    def set_output_statistics(self, mean, std):
        if mean.shape != self.output_mean.shape or std.shape != self.output_std.shape:
            raise ValueError("output statistics must match the decoder output dimension")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("output statistics must be finite")
        self.output_mean.copy_(mean.to(self.output_mean))
        self.output_std.copy_(std.to(self.output_std).clamp_min(1e-6))

    def normalize_output(self, output):
        return (output - self.output_mean) / self.output_std

    def forward(self, node_inputs, adjacency, z):
        node_inputs = node_inputs.float()
        N = node_inputs.size(0)
        
        # Broadcast global latent code z across all N mesh nodes
        z_expanded = z.expand(N, -1)
        x = torch.cat([node_inputs, z_expanded], dim=-1)
        x = F.relu(self.node_in_norm(self.node_in(x)))
        
        # Layer 1 Message Passing
        agg1 = aggregate_neighbors(adjacency, x)
        x = self.msg1_norm(F.relu(self.msg1(agg1)) + x)
        
        # Layer 2 Message Passing
        agg2 = aggregate_neighbors(adjacency, x)
        x = self.msg2_norm(F.relu(self.msg2(agg2)) + x)
        
        output_hidden = self.node_out[0](x)
        output_hidden = F.relu(self.node_out_norm(output_hidden))
        normalized_output = self.node_out[2](output_hidden)
        return normalized_output * self.output_std + self.output_mean


class GeometryToFlowGNN(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, latent_dim=16, out_dim=3):
        super().__init__()
        self.encoder = FlowInformedGeometryEncoder(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = FlowDecoderGNN(spatial_dim=in_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=out_dim)

    def forward(self, node_inputs, adjacency, obstacle_mask):
        node_inputs = node_inputs.float()
        z = self.encoder(node_inputs, adjacency, obstacle_mask)
        pred_velocity = self.decoder(node_inputs, adjacency, z)
        return pred_velocity, z

    def normalize_velocity(self, velocity):
        return self.decoder.normalize_output(velocity)


def _save_training_checkpoint(model, optimizer, epoch, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=checkpoint_path.parent,
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        torch.save(
            {
                "format_version": 1,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            temporary_path,
        )
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


from lbm_gnn import (
    LBMGraphSuperResolutionGNN,
    build_geometry_representations,
    load_lbm_superresolution_gnn,
    predict_fine_lbm_density,
    predict_fine_lbm_velocity,
    restrict_fine_lbm_velocity_residual,
    save_lbm_superresolution_gnn,
    train_lbm_superresolution_gnn,
)
from lbm_mesh import create_cubic_mesh_graph
from lbm_plotting import (
    plot_graph_flow,
    plot_graph_flow_comparison,
    plot_graph_mesh,
)
from lbm_solver import PyTorchGraphLBM3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    "LBMGraphSuperResolutionGNN",
    "PyTorchGraphLBM3D",
    "create_cubic_mesh_graph",
    "load_lbm_superresolution_gnn",
    "plot_graph_flow",
    "plot_graph_flow_comparison",
    "plot_graph_mesh",
    "predict_fine_lbm_velocity",
    "predict_fine_lbm_density",
    "restrict_fine_lbm_velocity_residual",
    "save_lbm_superresolution_gnn",
    "train_lbm_superresolution_gnn",
    "build_geometry_representations",
]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Domain Grid & Flow Parameters
    nx, ny, nz = 128, 64, 64
    training_radii = (6, 8, 10)
    test_radii = (6, 8, 10)
    reference_velocity = 0.05
    reference_density = 1.0
    stl_path = "Obstacle.stl"
    stl_fit = True
    stl_offset = (0.0, 0.0, 0.0)

    # Dataset splits
    training_geometries = ["sphere", "cylinder", "cube", "tetrahedron", "cone", "octahedron", "ellipsoid", "torus"]
    test_geometry = "dodecahedron"
    training_samples = [
        (geometry, sample_radius)
        for geometry in training_geometries
        for sample_radius in training_radii
    ]

    # Execution Settings
    lbm_warmup_steps = 600
    train_epochs = 100
    checkpoint_path = Path(__file__).resolve().with_name("cfd_flow_predictor_gnn.pt")
    mode = "resume"  # Use "train" to start fresh or "load" for inference only.

    print("=" * 60)
    print("CFD FLOW SURROGATE PREDICTION PIPELINE")
    print(f"Training Geometries: {training_geometries}")
    print(f"Training Radii: {training_radii}")
    print(f"Test Geometry: '{test_geometry}'")
    print(f"Test Radii: {test_radii}")
    print("=" * 60)

    # Instantiate Geometry-to-Flow Surrogate Model
    surrogate_model = GeometryToFlowGNN(
        in_dim=4,           # [x, y, z, obstacle_mask]
        hidden_dim=64,
        latent_dim=16,      # Learned flow-informed geometry vector dimension
        out_dim=3           # Predicts [u_x, u_y, u_z]
    ).to(device)

    # ---------------------------------------------------------
    # STEP 1: Train Encoder-Decoder End-to-End
    # ---------------------------------------------------------
    if mode in {"train", "resume"}:
        optimizer = torch.optim.AdamW(surrogate_model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        start_epoch = 1

        if mode == "resume":
            print(f"\nResuming training from '{checkpoint_path}'...")
            training_checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            required_keys = {
                "format_version",
                "epoch",
                "model_state_dict",
                "optimizer_state_dict",
            }
            if isinstance(training_checkpoint, dict) and required_keys.issubset(
                training_checkpoint
            ):
                if training_checkpoint["format_version"] != 1:
                    raise RuntimeError(
                        f"Checkpoint '{checkpoint_path}' uses an unsupported format version."
                    )
                completed_epoch = training_checkpoint["epoch"]
                if type(completed_epoch) is not int or completed_epoch < 0:
                    raise RuntimeError(
                        f"Checkpoint '{checkpoint_path}' has an invalid completed epoch."
                    )

                surrogate_model.load_state_dict(
                    training_checkpoint["model_state_dict"]
                )
                optimizer.load_state_dict(training_checkpoint["optimizer_state_dict"])
                start_epoch = completed_epoch + 1
                print(f"Restored through epoch {completed_epoch}.")
            else:
                surrogate_model.load_state_dict(training_checkpoint)
                _save_training_checkpoint(
                    surrogate_model, optimizer, 0, checkpoint_path
                )
                print(
                    "Loaded legacy model weights. This file has no saved optimizer "
                    "state or epoch number, so the optimizer starts fresh and training "
                    "continues from these weights at epoch 1."
                )

        def generate_training_sample(geom, sample_radius):
            mesh = create_cubic_mesh_graph(
                nx, ny, nz, sample_radius,
                geometry=geom,
                device=device,
                stl_path=stl_path,
                stl_fit=stl_fit,
                stl_offset=stl_offset,
            )
            (
                num_nodes,
                adjacency,
                obstacle_mask,
                inlet_mask,
                outlet_mask,
                outlet_upstream_idx,
                _positions,
            ) = mesh
            solver = PyTorchGraphLBM3D(
                num_nodes=num_nodes,
                adjacency=adjacency,
                obstacle_mask=obstacle_mask,
                inlet_mask=inlet_mask,
                outlet_mask=outlet_mask,
                outlet_upstream_idx=outlet_upstream_idx,
                tau=0.56,
                u_inlet=reference_velocity,
                reference_area=torch.pi * sample_radius**2,
                reference_velocity=reference_velocity,
                reference_density=reference_density,
                device=device,
            )
            with torch.no_grad():
                for _ in range(lbm_warmup_steps):
                    solver.step()
            return mesh, solver.u.clone()

        if not training_geometries:
            raise ValueError("training_geometries must contain at least one geometry")

        if mode == "train":
            velocity_sum = torch.zeros(3, device=device)
            velocity_squared_sum = torch.zeros(3, device=device)
            velocity_node_count = 0
            for sample_index, (geom, sample_radius) in enumerate(
                training_samples, start=1
            ):
                print(
                    "Calculating output normalization: "
                    f"sample {sample_index}/{len(training_samples)} "
                    f"({geom}, radius {sample_radius})",
                    flush=True,
                )
                _, training_velocity = generate_training_sample(geom, sample_radius)
                velocity_sum += training_velocity.sum(dim=1)
                velocity_squared_sum += training_velocity.square().sum(dim=1)
                velocity_node_count += training_velocity.size(1)

            output_mean = velocity_sum / velocity_node_count
            output_variance = (
                velocity_squared_sum / velocity_node_count - output_mean.square()
            ).clamp_min(0.0)
            output_std = output_variance.sqrt()
            surrogate_model.decoder.set_output_statistics(output_mean, output_std)
            _save_training_checkpoint(surrogate_model, optimizer, 0, checkpoint_path)

        print("\n[Phase 1/3] Training Encoder-Decoder GNN end-to-end...")
        for epoch in range(start_epoch, train_epochs + 1):
            total_epoch_loss = 0.0

            for sample_index, (geom, sample_radius) in enumerate(
                training_samples, start=1
            ):
                print(
                    f"Training epoch {epoch}/{train_epochs}: "
                    f"sample {sample_index}/{len(training_samples)} "
                    f"({geom}, radius {sample_radius})",
                    flush=True,
                )
                # 1. Build mesh graph and run LBM for the target flow
                mesh, gt_velocity = generate_training_sample(geom, sample_radius)
                (
                    num_nodes,
                    adjacency,
                    obstacle_mask,
                    inlet_mask,
                    outlet_mask,
                    outlet_upstream_idx,
                    positions,
                ) = mesh

                # 3. Model forward pass & loss step
                optimizer.zero_grad()
                # Build node_inputs with explicit float types
                node_inputs = torch.cat([positions.float(), obstacle_mask.float().unsqueeze(-1)], dim=-1)

                # Forward pass: calculates z, then predicts velocity field
                pred_velocity, z_latent = surrogate_model(node_inputs, adjacency, obstacle_mask)

                normalized_prediction = surrogate_model.normalize_velocity(pred_velocity)
                normalized_target = surrogate_model.normalize_velocity(gt_velocity.transpose(0, 1))
                loss = criterion(normalized_prediction, normalized_target)
                loss.backward()
                optimizer.step()

                total_epoch_loss += loss.item()

            avg_loss = total_epoch_loss / len(training_samples)
            _save_training_checkpoint(surrogate_model, optimizer, epoch, checkpoint_path)
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch [{epoch:02d}/{train_epochs:02d}] - Training MSE Loss: {avg_loss:.6e}")

        print(f"Training checkpoint saved to '{checkpoint_path}'.")

    elif mode == "load":
        print(f"\nLoading trained weights from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        load_result = surrogate_model.load_state_dict(state_dict, strict=False)
        legacy_normalization_keys = {"decoder.output_mean", "decoder.output_std"}
        missing_keys = set(load_result.missing_keys)
        if load_result.unexpected_keys or missing_keys - legacy_normalization_keys:
            raise RuntimeError(
                "Checkpoint does not match the model: "
                f"missing keys={load_result.missing_keys}, "
                f"unexpected keys={load_result.unexpected_keys}"
            )
        if missing_keys:
            print(
                "Checkpoint has no output normalization statistics; "
                "using identity normalization for legacy weights."
            )
    else:
        raise ValueError("mode must be 'train', 'resume', or 'load'")

    # ---------------------------------------------------------
    # STEP 2: Zero-Shot Prediction on Unseen Test Geometry
    # ---------------------------------------------------------
    surrogate_model.eval()
    for test_radius in test_radii:
        print(
            f"\n[Phase 2/3] Predicting '{test_geometry}' "
            f"with radius {test_radius}..."
        )
        test_mesh = create_cubic_mesh_graph(
            nx, ny, nz, test_radius,
            geometry=test_geometry,
            device=device,
            stl_path=stl_path,
            stl_fit=stl_fit,
            stl_offset=stl_offset,
        )
        (
            test_num_nodes,
            test_adjacency,
            test_obstacle_mask,
            test_inlet_mask,
            test_outlet_mask,
            test_outlet_upstream_idx,
            test_positions,
        ) = test_mesh

        with torch.no_grad():
            test_node_inputs = torch.cat(
                [test_positions, test_obstacle_mask.unsqueeze(-1)], dim=-1
            )
            predicted_test_velocity, test_z = surrogate_model(
                test_node_inputs, test_adjacency, test_obstacle_mask
            )

        print(
            f"Learned Latent Geometry Representation Vector for "
            f"'{test_geometry}' (radius {test_radius}):"
        )
        print(test_z.squeeze().cpu().numpy())

        # ---------------------------------------------------------
        # STEP 3: Validate against LBM Ground Truth
        # ---------------------------------------------------------
        print(
            f"\n[Phase 3/3] Solving LBM ground truth for "
            f"'{test_geometry}' (radius {test_radius})..."
        )
        test_ref_area = torch.pi * test_radius**2
        test_gt_solver = PyTorchGraphLBM3D(
            num_nodes=test_num_nodes,
            adjacency=test_adjacency,
            obstacle_mask=test_obstacle_mask,
            inlet_mask=test_inlet_mask,
            outlet_mask=test_outlet_mask,
            outlet_upstream_idx=test_outlet_upstream_idx,
            tau=0.56,
            u_inlet=reference_velocity,
            reference_area=test_ref_area,
            reference_velocity=reference_velocity,
            reference_density=reference_density,
            device=device,
        )

        sim_start = time.perf_counter()
        for _ in range(lbm_warmup_steps):
            test_gt_solver.step()

        gt_test_velocity = test_gt_solver.u
        rel_l2_error = (
            torch.norm(predicted_test_velocity.transpose(0, 1) - gt_test_velocity)
            / torch.norm(gt_test_velocity)
        ).item()

        print(f"Ground truth CFD finished in {time.perf_counter() - sim_start:.2f} s.")
        print(
            f"--> Zero-Shot Generalization Relative L2 Velocity Error "
            f"(radius {test_radius}): {rel_l2_error * 100:.2f}%"
        )

        plot_graph_flow(
            test_gt_solver,
            test_positions,
            nx, ny, nz,
            geometry=f"{test_geometry} (radius {test_radius})",
            velocity=predicted_test_velocity.transpose(0, 1),
        )
        plot_graph_flow(
            test_gt_solver,
            test_positions,
            nx, ny, nz,
            geometry=f"{test_geometry} (radius {test_radius})",
            velocity=gt_test_velocity,
        )