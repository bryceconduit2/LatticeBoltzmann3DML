import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator


def _plot_3d_streamlines(ax, velocity, positions, obstacle, nx, ny, nz, colormap, norm, seed_spacing=4, step_size=0.5, max_steps=400):
    """Integrate and draw streamlines through a regular node-centered velocity field."""
    pos = positions.cpu().numpy()
    velocity = velocity.detach().cpu().numpy().T
    components = []
    for component in range(3):
        values = np.zeros((nz, ny, nx), dtype=float)
        for point, value in zip(pos.astype(int), velocity[:, component]):
            x, y, z = point
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                values[z, y, x] = value
        components.append(RegularGridInterpolator(
            (np.arange(nz), np.arange(ny), np.arange(nx)),
            values,
            bounds_error=False,
            fill_value=0.0,
        ))

    def velocity_at(point):
        z, y, x = point[2], point[1], point[0]
        return np.array([component((z, y, x)) for component in components], dtype=float)

    def is_fluid(point):
        x, y, z = np.rint(point).astype(int)
        return 0 <= x < nx and 0 <= y < ny and 0 <= z < nz and not obstacle.reshape(nz, ny, nx)[z, y, x]

    seeds = []
    for y in range(seed_spacing // 2, ny, seed_spacing):
        for z in range(seed_spacing // 2, nz, seed_spacing):
            seed = np.array([0.5, float(y), float(z)])
            if is_fluid(seed):
                seeds.append(seed)

    for seed in seeds:
        points = [seed]
        point = seed.copy()
        for _ in range(max_steps):
            if not is_fluid(point):
                break
            local_velocity = velocity_at(point)
            speed = np.linalg.norm(local_velocity)
            if speed < 1e-10:
                break
            point = point + step_size * local_velocity / speed
            if not is_fluid(point):
                break
            points.append(point.copy())
        if len(points) > 1:
            line = np.asarray(points)
            speed = np.linalg.norm(np.array([velocity_at(point) for point in line]), axis=1)
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=colormap(norm(speed.mean())), linewidth=1.0, alpha=0.85)


def plot_graph_flow(solver, positions, nx, ny, nz, geometry="cylinder", plot_all=False, show_obstacles=True, plot_obstacle_all=True, debug=False, velocity=None, plot_mode="slices", streamline_seed_spacing=4):
    if plot_mode not in {"slices", "streamlines"}:
        raise ValueError("plot_mode must be 'slices' or 'streamlines'")
    plotted_velocity = solver.u if velocity is None else velocity
    pressure_force = solver.pressure_force().detach().cpu().numpy()
    drag = pressure_force[0]
    drag_coefficient = solver.drag_coefficient().item()
    speed = torch.linalg.vector_norm(plotted_velocity, dim=0).detach().cpu().numpy()
    pos = positions.cpu().numpy()
    obs = solver.obstacle.cpu().numpy()
    speed_max = float(np.nanmax(speed))
    slice_z = [nz // 4, nz // 2, (3 * nz) // 4]

    if debug:
        fluid = ~obs
        print(f"plot_graph_flow: points={len(pos)}, fluid={int(fluid.sum())}, obstacle={int(obs.sum())}, x_range=({pos[:, 0].min()}, {pos[:, 0].max()}), speed_range=({speed.min():.3e}, {speed.max():.3e})")
        for x_value in (0, nx // 4, nx // 2, (3 * nx) // 4, nx - 1):
            x_fluid_speed = speed[(pos[:, 0] == x_value) & fluid]
            print(f"  x={x_value}: fluid_points={len(x_fluid_speed)}, speed_mean={x_fluid_speed.mean():.3e}")

    norm = Normalize(vmin=0.0, vmax=max(speed_max, 1e-8))
    colormap = plt.get_cmap("turbo")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    contour = None

    if plot_mode == "streamlines":
        _plot_3d_streamlines(ax, plotted_velocity, positions, obs, nx, ny, nz, colormap, norm, seed_spacing=streamline_seed_spacing)
        if show_obstacles:
            ax.scatter(pos[obs, 0], pos[obs, 1], pos[obs, 2], color="black", marker="s", s=8, alpha=0.8, edgecolors="none")
    elif plot_all:
        fluid = ~obs
        ax.scatter(pos[fluid, 0], pos[fluid, 1], pos[fluid, 2], c=speed[fluid], cmap=colormap, norm=norm, marker="s", s=8, alpha=0.9, edgecolors="none")
        ax.scatter(pos[obs, 0], pos[obs, 1], pos[obs, 2], color="black", marker="s", s=8, alpha=1.0, edgecolors="none")
    elif plot_obstacle_all and show_obstacles:
        ax.scatter(pos[obs, 0], pos[obs, 1], pos[obs, 2], color="black", marker="s", s=8, alpha=1.0, edgecolors="none")

    if plot_mode == "slices" and not plot_all:
        x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
        for z_val in slice_z:
            mask = pos[:, 2] == z_val
            slice_speed = np.ma.array(speed[mask].reshape(ny, nx), mask=obs[mask].reshape(ny, nx))
            contour = ax.contourf(x_grid, y_grid, slice_speed, levels=30, cmap=colormap, norm=norm, alpha=0.9, zdir="z", offset=z_val)
            if show_obstacles and not plot_obstacle_all:
                slice_pos = pos[mask]
                slice_obs = obs[mask]
                ax.scatter(slice_pos[slice_obs, 0], slice_pos[slice_obs, 1], slice_pos[slice_obs, 2], color="black", marker="s", s=15, alpha=1.0, edgecolors="none")

    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
    ax.set_xlabel("x: flow direction"); ax.set_ylabel("y"); ax.set_zlabel("z: depth")
    view_name = "3D streamlines" if plot_mode == "streamlines" else ("all mesh points" if plot_all else "three z-slices")
    ax.set_title(
        f"Graph-based D3Q19 LBM: {geometry} - {view_name}\n"
        f"Pressure drag: {drag:.6e} (lattice units)\n"
        f"Cd: {drag_coefficient:.6e}"
    )
    ax.view_init(elev=28, azim=-62)
    if contour is not None:
        fig.subplots_adjust(right=0.84)
        fig.colorbar(contour, ax=ax, pad=0.12, fraction=0.045, label="Velocity magnitude |U|")
    plt.show()


def plot_graph_flow_comparison(coarse_solver, fine_solver, superres_velocity, superres_density, coarse_positions, fine_positions, coarse_nx, coarse_ny, coarse_nz, fine_nx, fine_ny, fine_nz, geometry="cylinder", plot_mode="slices", streamline_seed_spacing=4):
    """Compare coarse, conventional fine, and GNN-refined fine fields."""
    if plot_mode not in {"slices", "streamlines"}:
        raise ValueError("plot_mode must be 'slices' or 'streamlines'")
    coarse_speed = torch.linalg.vector_norm(coarse_solver.u, dim=0).detach().cpu().numpy()
    fine_speed = torch.linalg.vector_norm(fine_solver.u, dim=0).detach().cpu().numpy()
    superres_speed = torch.linalg.vector_norm(superres_velocity, dim=0).detach().cpu().numpy()
    coarse_drag = coarse_solver.pressure_force()[0].item()
    coarse_cd = coarse_solver.drag_coefficient().item()
    fine_drag = fine_solver.pressure_force()[0].item()
    fine_cd = fine_solver.drag_coefficient().item()
    superres_drag = fine_solver.pressure_force(density=superres_density)[0].item()
    superres_cd = fine_solver.drag_coefficient(density_field=superres_density).item()
    norm = Normalize(
        vmin=0.0,
        vmax=max(
            float(np.nanmax(coarse_speed)), float(np.nanmax(fine_speed)),
            float(np.nanmax(superres_speed)), 1e-8,
        ),
    )
    colormap = plt.get_cmap("turbo")
    figure = plt.figure(figsize=(22, 8))
    axes = [
        figure.add_subplot(131, projection="3d"),
        figure.add_subplot(132, projection="3d"),
        figure.add_subplot(133, projection="3d"),
    ]
    panel_titles = (
        f"Coarse LBM\ndrag={coarse_drag:.6e}\nCd={coarse_cd:.6e}",
        f"Conventional fine LBM\ndrag={fine_drag:.6e}\nCd={fine_cd:.6e}",
        f"GNN super-resolution\ndrag={superres_drag:.6e}\nCd={superres_cd:.6e}",
    )
    panel_velocities = (coarse_solver.u, fine_solver.u, superres_velocity)
    panel_speeds = (coarse_speed, fine_speed, superres_speed)
    panel_positions = (coarse_positions, fine_positions, fine_positions)
    panel_obstacles = (coarse_solver.obstacle, fine_solver.obstacle, fine_solver.obstacle)
    panel_shapes = (
        (coarse_nx, coarse_ny, coarse_nz),
        (fine_nx, fine_ny, fine_nz),
        (fine_nx, fine_ny, fine_nz),
    )
    speed_contour = None
    for panel_index, (axis, velocity, speed, title) in enumerate(
        zip(axes, panel_velocities, panel_speeds, panel_titles)
    ):
        positions = panel_positions[panel_index]
        obstacle = panel_obstacles[panel_index].cpu().numpy()
        panel_nx, panel_ny, panel_nz = panel_shapes[panel_index]
        pos = positions.cpu().numpy()
        if plot_mode == "streamlines":
            _plot_3d_streamlines(
                axis, velocity, positions, obstacle, panel_nx, panel_ny, panel_nz,
                colormap, norm, seed_spacing=streamline_seed_spacing,
            )
        else:
            slice_z = [panel_nz // 4, panel_nz // 2, (3 * panel_nz) // 4]
            x_grid, y_grid = np.meshgrid(np.arange(panel_nx), np.arange(panel_ny), indexing="xy")
            for z_val in slice_z:
                mask = pos[:, 2] == z_val
                slice_speed = np.ma.array(
                    speed[mask].reshape(panel_ny, panel_nx),
                    mask=obstacle[mask].reshape(panel_ny, panel_nx),
                )
                contour = axis.contourf(x_grid, y_grid, slice_speed, levels=30, cmap=colormap, norm=norm, alpha=0.9, zdir="z", offset=z_val)
                speed_contour = contour
        axis.scatter(pos[obstacle, 0], pos[obstacle, 1], pos[obstacle, 2], color="black", marker="s", s=8, alpha=1.0, edgecolors="none")
        axis.set_xlim(0, panel_nx); axis.set_ylim(0, panel_ny); axis.set_zlim(0, panel_nz); axis.set_box_aspect((panel_nx, panel_ny, panel_nz))
        view_name = "3D streamlines" if plot_mode == "streamlines" else "three z-slices"
        axis.set_xlabel("x: flow direction"); axis.set_ylabel("y"); axis.set_zlabel("z"); axis.set_title(f"{title} - {view_name}"); axis.view_init(elev=28, azim=-62)
    figure.subplots_adjust(right=0.86)
    speed_colorbar = speed_contour if speed_contour is not None else plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    speed_colorbar.set_array([])
    figure.colorbar(speed_colorbar, ax=axes, pad=0.08, fraction=0.035, label="Velocity magnitude")
    figure.suptitle(
        f"Coarse vs conventional fine vs GNN super-resolution\n{geometry}"
    )
    plt.show()


def plot_graph_mesh(positions, obstacle_mask, nx, ny, nz, geometry="cylinder", z_slices=None):
    """Plot the generated graph domain and obstacle without running the solver."""
    pos = positions.cpu().numpy(); obstacle = obstacle_mask.cpu().numpy()
    fig = plt.figure(figsize=(12, 8)); ax = fig.add_subplot(111, projection="3d")
    if z_slices is None:
        fluid = ~obstacle
        ax.scatter(pos[fluid, 0], pos[fluid, 1], pos[fluid, 2], color="steelblue", marker=".", s=3, alpha=0.35, edgecolors="none")
        ax.scatter(pos[obstacle, 0], pos[obstacle, 1], pos[obstacle, 2], color="black", marker=".", s=6, alpha=1.0, edgecolors="none")
        title = f"Generated {geometry} mesh: all nodes"
    else:
        for z_val in z_slices:
            mask = pos[:, 2] == z_val; slice_pos = pos[mask]; slice_obstacle = obstacle[mask]
            fluid = ~slice_obstacle
            ax.scatter(slice_pos[fluid, 0], slice_pos[fluid, 1], slice_pos[fluid, 2], color="lightgray", marker="s", s=10, alpha=0.25, edgecolors="none")
            ax.scatter(slice_pos[slice_obstacle, 0], slice_pos[slice_obstacle, 1], slice_pos[slice_obstacle, 2], color="black", marker="s", s=18, alpha=1.0, edgecolors="none")
        title = f"Generated {geometry} mesh: selected z-slices"
    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
    ax.set_xlabel("x: flow direction"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.set_title(title); ax.view_init(elev=28, azim=-62)
    plt.tight_layout(); plt.show()
