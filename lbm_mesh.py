import numpy as np
import torch
import trimesh
import math

def _create_stl_obstacle_mask(
    stl_path, x_flat, y_flat, z_flat, nx, ny, nz, radius, stl_fit=True,
    stl_offset=(0.0, 0.0, 0.0),
):
    """Load an STL and classify lattice nodes inside its closed surface."""
    mesh = trimesh.load_mesh(stl_path, force="mesh", process=True)
    if mesh.is_empty:
        raise ValueError(f"STL mesh is empty: {stl_path}")
    if not mesh.is_watertight:
        raise ValueError("STL mesh must be closed/watertight for volume classification")

    if stl_fit:
        largest_extent = float(np.max(mesh.bounding_box.extents))
        if largest_extent <= 0.0:
            raise ValueError("STL mesh has zero size")
        mesh.apply_scale((2.0 * radius) / largest_extent)

    target_center = np.array([nx / 4.0, ny / 2.0, nz / 2.0])
    mesh.apply_translation(target_center + np.asarray(stl_offset) - mesh.bounding_box.centroid)
    points = torch.stack([x_flat, y_flat, z_flat], dim=1).detach().cpu().numpy()
    return torch.as_tensor(
        mesh.contains(points), dtype=torch.bool, device=x_flat.device
    )

def create_cubic_mesh_graph(
    nx,
    ny,
    nz,
    radius,
    geometry="cylinder",
    device=None,
    stl_path=None,
    stl_fit=True,
    stl_offset=(0.0, 0.0, 0.0),
):
    """Create graph topology and masks for built-in polyhedra and solids, or an STL geometry.

    The cone points along +z. The ellipsoid has semi-axes
    (radius, 0.75 * radius, 0.5 * radius). The torus lies in the xy-plane
    with a major radius of radius and a tube radius of radius / 3.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_geometries = {
        "cylinder",
        "sphere",
        "cube",
        "tetrahedron",
        "dodecahedron",
        "cone",
        "octahedron",
        "ellipsoid",
        "torus",
        "stl",
    }
    if geometry not in valid_geometries:
        raise ValueError(f"geometry must be one of {valid_geometries}")
    if geometry == "stl" and stl_path is None:
        raise ValueError("stl_path is required when geometry='stl'")

    num_nodes = nx * ny * nz
    z_grid, y_grid, x_grid = torch.meshgrid(
        torch.arange(nz, device=device),
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    z_flat, y_flat, x_flat = (
        z_grid.flatten(),
        y_grid.flatten(),
        x_grid.flatten(),
    )
    positions = torch.stack([x_flat, y_flat, z_flat], dim=1)

    directions = [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 0],
        [-1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [1, 0, 1],
        [-1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [0, 1, 1],
        [0, -1, 1],
        [0, -1, -1],
        [0, 1, -1],
    ]
    adjacency = torch.zeros((19, num_nodes), dtype=torch.long, device=device)
    for direction, (dx, dy, dz) in enumerate(directions):
        x_prev = (x_flat - dx) % nx
        y_prev = (y_flat - dy) % ny
        z_prev = (z_flat - dz) % nz
        adjacency[direction] = z_prev * (ny * nx) + y_prev * nx + x_prev

    cx, cy = nx // 4, ny // 2
    cz = nz // 2

    if geometry == "sphere":
        obstacle_mask = (
            (x_flat - cx) ** 2 + (y_flat - cy) ** 2 + (z_flat - cz) ** 2
        ) <= radius**2

    elif geometry == "cylinder":
        obstacle_mask = ((x_flat - cx) ** 2 + (y_flat - cy) ** 2) <= radius**2

    elif geometry == "cube":
        obstacle_mask = (
            (torch.abs(x_flat - cx) <= radius)
            & (torch.abs(y_flat - cy) <= radius)
            & (torch.abs(z_flat - cz) <= radius)
        )

    elif geometry == "cone":
        dx, dy, dz = x_flat - cx, y_flat - cy, z_flat - cz
        cone_radius = (radius - dz) / 2.0
        obstacle_mask = (
            (torch.abs(dz) <= radius)
            & (dx**2 + dy**2 <= cone_radius**2)
        )

    elif geometry == "octahedron":
        dx, dy, dz = x_flat - cx, y_flat - cy, z_flat - cz
        obstacle_mask = (
            torch.abs(dx) + torch.abs(dy) + torch.abs(dz)
        ) <= radius

    elif geometry == "ellipsoid":
        dx, dy, dz = x_flat - cx, y_flat - cy, z_flat - cz
        obstacle_mask = (
            dx**2 * (0.75 * 0.5) ** 2
            + dy**2 * 0.5**2
            + dz**2 * 0.75**2
        ) <= (radius * 0.75 * 0.5) ** 2

    elif geometry == "torus":
        dx, dy, dz = x_flat - cx, y_flat - cy, z_flat - cz
        radial_distance = torch.sqrt(dx**2 + dy**2)
        tube_radius = radius / 3.0
        obstacle_mask = (
            (radial_distance - radius) ** 2 + dz**2 <= tube_radius**2
        )

    elif geometry == "tetrahedron":
        dx, dy, dz = x_flat - cx, y_flat - cy, z_flat - cz
        scale = radius / math.sqrt(3)

        c1 = (-dx - dy - dz) <= scale
        c2 = (-dx + dy + dz) <= scale
        c3 = (dx - dy + dz) <= scale
        c4 = (dx + dy - dz) <= scale
        obstacle_mask = c1 & c2 & c3 & c4

    elif geometry == "dodecahedron":
        # Regular dodecahedron centered at (cx, cy, cz) with circumradius = radius
        dx, dy, dz = x_flat - cx, y_flat - cy, z_flat - cz
        abs_x, abs_y, abs_z = torch.abs(dx), torch.abs(dy), torch.abs(dz)

        phi = (1.0 + math.sqrt(5.0)) / 2.0  # Golden ratio
        threshold = ((phi + 1.0) / math.sqrt(3.0)) * radius

        # 12 face planes derived from golden ratio symmetries
        c1 = abs_y + phi * abs_z <= threshold
        c2 = abs_z + phi * abs_x <= threshold
        c3 = abs_x + phi * abs_y <= threshold
        obstacle_mask = c1 & c2 & c3

    else:
        obstacle_mask = _create_stl_obstacle_mask(
            stl_path,
            x_flat,
            y_flat,
            z_flat,
            nx,
            ny,
            nz,
            radius,
            stl_fit=stl_fit,
            stl_offset=stl_offset,
        )

    inlet_mask = x_flat == 0
    outlet_mask = x_flat == nx - 1
    outlet_upstream_idx = (
        z_flat[outlet_mask] * (ny * nx) + y_flat[outlet_mask] * nx + (nx - 2)
    )

    return (
        num_nodes,
        adjacency,
        obstacle_mask,
        inlet_mask,
        outlet_mask,
        outlet_upstream_idx,
        positions,
    )