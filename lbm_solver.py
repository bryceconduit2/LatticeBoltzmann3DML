import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyTorchGraphLBM3D:
    """Graph-based D3Q19 lattice-Boltzmann flow for arbitrary geometries."""

    def __init__(
        self,
        num_nodes,
        adjacency,
        obstacle_mask,
        inlet_mask,
        outlet_mask,
        outlet_upstream_idx,
        tau=0.6,
        u_inlet=0.04,
        reference_area=None,
        reference_velocity=None,
        reference_density=1.0,
        device=device,
    ):
        self.num_nodes = num_nodes
        self.tau = tau
        self.u_inlet = u_inlet
        self.reference_area = reference_area
        self.reference_velocity = u_inlet if reference_velocity is None else reference_velocity
        self.reference_density = reference_density
        self.device = device
        self.population_floor = 1e-12
        self.step_count = 0

        self.e = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1],
                [1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0],
                [1, 0, 1], [-1, 0, 1], [-1, 0, -1], [1, 0, -1],
                [0, 1, 1], [0, -1, 1], [0, -1, -1], [0, 1, -1],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.w = torch.tensor(
            [1 / 3, *([1 / 18] * 6), *([1 / 36] * 12)],
            dtype=torch.float32,
            device=device,
        )
        self.opposite = [0, 2, 1, 4, 3, 6, 5, 9, 10, 7, 8, 13, 14, 11, 12, 17, 18, 15, 16]

        self.adjacency = adjacency.to(device)
        self.obstacle = obstacle_mask.to(device)
        self.inlet = inlet_mask.to(device)
        self.outlet = outlet_mask.to(device)
        self.outlet_upstream = outlet_upstream_idx.to(device)

        self.rho = torch.ones(num_nodes, dtype=torch.float32, device=device)
        self.u = torch.zeros((3, num_nodes), dtype=torch.float32, device=device)
        self.u[0].fill_(u_inlet)
        self.u[:, self.obstacle] = 0.0
        self.f = self.compute_equilibrium(self.rho, self.u)

    def compute_equilibrium(self, rho, u):
        f_eq = torch.empty((19, self.num_nodes), dtype=rho.dtype, device=self.device)
        u_sq = torch.sum(u * u, dim=0)
        for direction in range(19):
            e_dot_u = (
                self.e[direction, 0] * u[0]
                + self.e[direction, 1] * u[1]
                + self.e[direction, 2] * u[2]
            )
            f_eq[direction] = self.w[direction] * rho * (
                1.0 + 3.0 * e_dot_u + 4.5 * e_dot_u**2 - 1.5 * u_sq
            )
        return f_eq

    @torch.no_grad()
    def inject_velocity(self, velocity, blend=1.0):
        """Blend in a velocity field while preserving the current density state."""
        if velocity.shape != self.u.shape:
            raise ValueError(
                f"velocity must have shape {tuple(self.u.shape)}, got {tuple(velocity.shape)}"
            )
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be between 0 and 1")
        velocity = velocity.to(device=self.device, dtype=self.u.dtype).clone()
        velocity = self.u + blend * (velocity - self.u)
        velocity[:, self.inlet] = 0.0
        velocity[0, self.inlet] = self.u_inlet
        velocity[:, self.obstacle] = 0.0
        self.f += self.compute_equilibrium(self.rho, velocity) - self.compute_equilibrium(self.rho, self.u)
        self.u.copy_(velocity)

    def pressure_force(self, reference_density=None, density=None):
        """Return the pressure force exerted on the obstacle in lattice units."""
        pressure_density = self.rho if density is None else density
        if reference_density is None:
            outlet_fluid = self.outlet & ~self.obstacle
            reference_density = pressure_density[outlet_fluid].mean()

        force = torch.zeros(3, dtype=self.rho.dtype, device=self.device)
        for direction in range(1, 19):
            obstacle_nodes = self.obstacle
            fluid_neighbors = ~self.obstacle[self.adjacency[direction]]
            link_mask = obstacle_nodes & fluid_neighbors
            if link_mask.any():
                source_nodes = self.adjacency[direction, link_mask]
                pressure = (pressure_density[source_nodes] - reference_density) / 3.0
                force += (3.0 * self.w[direction] * pressure).sum() * self.e[direction]
        return force

    def drag_coefficient(
        self, reference_area=None, reference_velocity=None,
        reference_density=None, density_field=None,
    ):
        """Return pressure drag coefficient using lattice-unit reference values."""
        area = self.reference_area if reference_area is None else reference_area
        velocity = self.reference_velocity if reference_velocity is None else reference_velocity
        density = self.reference_density if reference_density is None else reference_density
        if area is None or area <= 0:
            raise ValueError("reference_area must be positive to calculate drag coefficient")
        if velocity <= 0 or density <= 0:
            raise ValueError("reference_velocity and reference_density must be positive")
        drag = self.pressure_force(reference_density=density, density=density_field)[0]
        return 2.0 * drag / (density * velocity**2 * area)

    def step(self):
        self.f = torch.nan_to_num(
            self.f,
            nan=self.population_floor,
            posinf=1.0,
            neginf=self.population_floor,
        ).clamp_min(self.population_floor)
        self.rho = torch.sum(self.f, dim=0).clamp_min(1e-8)
        for axis in range(3):
            self.u[axis] = torch.sum(
                self.f * self.e[:, axis].unsqueeze(1), dim=0
            ) / self.rho

        self.u[0, self.inlet] = self.u_inlet
        self.u[1:, self.inlet] = 0.0
        self.u[:, self.obstacle] = 0.0

        equilibrium = self.compute_equilibrium(self.rho, self.u)
        population_delta = self.f - equilibrium
        base_relaxation = 1.0 / self.tau
        unsafe = population_delta > 0.0
        relaxation_limit = torch.where(
            unsafe,
            (self.f - self.population_floor)
            / population_delta.clamp_min(1e-30),
            torch.full_like(self.f, float("inf")),
        ).amin(dim=0)
        relaxation = torch.minimum(
            torch.full_like(relaxation_limit, base_relaxation),
            0.95 * relaxation_limit,
        )
        f_post_collision = self.f - population_delta * relaxation.unsqueeze(0)
        f_post_collision = f_post_collision.clamp_min(self.population_floor)

        for direction in range(19):
            f_post_collision[direction, self.obstacle] = f_post_collision[
                self.opposite[direction], self.obstacle
            ]

        for direction in range(19):
            self.f[direction] = f_post_collision[direction, self.adjacency[direction]]

        self.f[:, self.outlet] = self.f[:, self.outlet_upstream]

        self.step_count += 1
