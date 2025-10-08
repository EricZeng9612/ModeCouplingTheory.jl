"""Binary mixture overdamped MCT solver in 2D.

This module adapts the numerical ideas implemented in the Julia package
``ModeCouplingTheory.jl`` to a Python workflow tailored for a 2D Brownian
binary Lennard-Jones fluid.  It provides the following key features:

* Static input from a two-component Lennard-Jones mixture evaluated within a
  random-phase approximation.
* Memory kernels built with the standard ``q-k`` convolution structure.
* Flexible angular integration that can either rely on an analytic change of
  variables or on a high-resolution Gauss-Legendre quadrature in ``θ``.
* Predictor-corrector time stepping with an iterative non-linear feedback loop
  at every step until convergence is reached.
* Normalised correlators ``φ_norm = Φ / S`` provided transparently to the
  caller.

The implementation emphasises clarity and extensibility so that the individual
components can be inspected and modified easily while still providing a
practical solver for medium sized grids.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray


Array2x2 = NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class Species:
    """Single species specification for a Lennard-Jones binary mixture."""

    name: str
    density: float
    mass: float


@dataclasses.dataclass(frozen=True)
class LennardJonesParameters:
    """Parameters controlling the Lennard-Jones pair interactions."""

    epsilon: Dict[Tuple[str, str], float]
    sigma: Dict[Tuple[str, str], float]
    r_cutoff: float = 5.0

    def epsilon_for(self, a: str, b: str) -> float:
        return self.epsilon[(a, b)] if (a, b) in self.epsilon else self.epsilon[(b, a)]

    def sigma_for(self, a: str, b: str) -> float:
        return self.sigma[(a, b)] if (a, b) in self.sigma else self.sigma[(b, a)]


class RadialInterpolator:
    """One-dimensional interpolator for radial, matrix-valued data."""

    def __init__(self, q: NDArray[np.float64], values: NDArray[np.float64]) -> None:
        self.q = q
        self.values = values

    def __call__(self, q_val: float) -> NDArray[np.float64]:
        q_clipped = np.clip(q_val, self.q[0], self.q[-1])
        result = np.empty_like(self.values[0])
        for i in range(self.values.shape[1]):
            for j in range(self.values.shape[2]):
                result[i, j] = np.interp(q_clipped, self.q, self.values[:, i, j])
        return result


def approximate_lj_transform(q: float, epsilon: float, sigma: float) -> float:
    """Approximate 2D Fourier transform of the Lennard-Jones potential.

    The expression is based on a Gaussian screening of the Lennard-Jones core
    and reproduces the qualitative behaviour of the true transform.  This keeps
    the implementation lightweight while still providing a smooth input for the
    mode-coupling functional.
    """

    sigma2 = sigma * sigma
    return -2.0 * math.pi * epsilon * sigma2 * math.exp(-0.5 * (q * sigma) ** 2)


class AngularIntegrator:
    """Angular integration helper supporting analytic and quadrature modes."""

    def __init__(self, method: str = "quadrature", n_points: int = 256) -> None:
        method = method.lower()
        if method not in {"analytic", "quadrature"}:
            raise ValueError(f"Unknown angular integration method: {method}")
        self.method = method
        self.n_points = n_points
        if method == "quadrature":
            nodes, weights = np.polynomial.legendre.leggauss(n_points)
            # Map from [-1, 1] to [0, 2π].
            self.theta_nodes = 0.5 * (nodes + 1.0) * (2.0 * np.pi)
            self.theta_weights = weights * np.pi
        else:
            nodes, weights = np.polynomial.legendre.leggauss(n_points)
            self.u_nodes = nodes
            self.u_weights = weights

    def integrate(
        self,
        q: float,
        k: float,
        integrand: Callable[[float, float], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        if self.method == "quadrature":
            acc = None
            for theta, weight in zip(self.theta_nodes, self.theta_weights):
                cos_theta = math.cos(theta)
                q_minus_k = math.sqrt(max(q * q + k * k - 2.0 * q * k * cos_theta, 1e-16))
                value = integrand(cos_theta, q_minus_k)
                acc = value * weight if acc is None else acc + value * weight
            assert acc is not None
            return acc
        # Analytic change-of-variable: integrate over p = |q - k|.
        p_min = abs(q - k)
        p_max = q + k
        if p_max <= 1e-12:
            return integrand(1.0, 0.0) * 2.0 * np.pi
        half_range = 0.5 * (p_max - p_min)
        center = 0.5 * (p_max + p_min)
        acc = None
        for u, weight in zip(self.u_nodes, self.u_weights):
            p = center + half_range * u
            # Recover cos(theta) from the law of cosines.
            cos_theta = (q * q + k * k - p * p) / (2.0 * q * k + 1e-18)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            sin_theta = math.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
            jacobian = (p / (q * k + 1e-18)) / max(sin_theta, 1e-12)
            weight_p = weight * half_range * 2.0  # Account for 0..2π symmetry.
            value = integrand(cos_theta, p) * weight_p * jacobian
            acc = value if acc is None else acc + value
        assert acc is not None
        return acc


class StaticStructureBuilder:
    """Construct the static structure factor matrix for the mixture."""

    def __init__(
        self,
        species: Tuple[Species, Species],
        lj_params: LennardJonesParameters,
        temperature: float,
    ) -> None:
        self.species = species
        self.lj_params = lj_params
        self.temperature = temperature

    def build(self, q_grid: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        beta = 1.0 / self.temperature
        n_q = len(q_grid)
        c_q = np.zeros((n_q, 2, 2), dtype=np.float64)
        for idx, q in enumerate(q_grid):
            for i, a in enumerate(self.species):
                for j, b in enumerate(self.species):
                    eps = self.lj_params.epsilon_for(a.name, b.name)
                    sig = self.lj_params.sigma_for(a.name, b.name)
                    c_q[idx, i, j] = -beta * approximate_lj_transform(q, eps, sig)
        sqrt_rho = np.diag([math.sqrt(s.density) for s in self.species])
        identity = np.eye(2)
        s_q = np.zeros_like(c_q)
        for idx in range(n_q):
            kernel = identity - sqrt_rho @ c_q[idx] @ sqrt_rho
            s_q[idx] = np.linalg.inv(kernel)
        return s_q, c_q


class MemoryKernelBuilder:
    """Assemble the mode-coupling memory kernel for each wave vector."""

    def __init__(
        self,
        q_grid: NDArray[np.float64],
        static_structure: NDArray[np.float64],
        direct_correlation: NDArray[np.float64],
        species: Tuple[Species, Species],
        integrator: AngularIntegrator,
    ) -> None:
        self.q_grid = q_grid
        self.static_structure = static_structure
        self.direct_correlation = direct_correlation
        self.species = species
        self.integrator = integrator
        self.c_interp = RadialInterpolator(q_grid, direct_correlation)

    def _vertex(
        self,
        q: float,
        k: float,
        cos_theta: float,
        q_minus_k: float,
    ) -> NDArray[np.float64]:
        vertex = np.zeros((2, 2), dtype=np.float64)
        c_k = self.c_interp(k)
        c_qmk = self.c_interp(q_minus_k)
        for a in range(2):
            for alpha in range(2):
                term1 = k * cos_theta * c_k[a, alpha]
                term2 = (q - k * cos_theta) * c_qmk[a, alpha]
                vertex[a, alpha] = term1 + term2
        return vertex

    def evaluate(
        self,
        q_index: int,
        phi_snapshot: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        q = self.q_grid[q_index]
        s_inv = np.linalg.inv(self.static_structure[q_index])
        dq = np.gradient(self.q_grid)
        prefactor = 1.0 / (2.0 * math.pi) ** 2
        result = np.zeros((2, 2), dtype=np.float64)
        for k_idx, k in enumerate(self.q_grid):
            def integrand(cos_theta: float, q_minus_k: float) -> NDArray[np.float64]:
                vertex_a = self._vertex(q, k, cos_theta, q_minus_k)
                vertex_b = vertex_a  # Symmetric for density correlators.
                phi_k = phi_snapshot[k_idx]
                phi_qmk = self._interpolate_phi(q_minus_k, phi_snapshot)
                contribution = vertex_a @ phi_k @ s_inv @ vertex_b.T
                contribution = contribution * np.trace(phi_qmk @ s_inv)
                return contribution

            angular_integral = self.integrator.integrate(q, k, integrand)
            weight = k * dq[k_idx]
            result += angular_integral * weight
        return prefactor * result

    def _interpolate_phi(
        self,
        q_val: float,
        phi_snapshot: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        q_clipped = np.clip(q_val, self.q_grid[0], self.q_grid[-1])
        interp = np.empty((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                interp[i, j] = np.interp(q_clipped, self.q_grid, phi_snapshot[:, i, j])
        return interp


class PredictorCorrectorSolver:
    """Time integrator for the binary MCT equations."""

    def __init__(
        self,
        q_grid: NDArray[np.float64],
        static_structure: NDArray[np.float64],
        diffusion_constants: NDArray[np.float64],
        kernel_builder: MemoryKernelBuilder,
        dt: float,
        max_iter: int = 30,
        tol: float = 1e-6,
        mixing: float = 0.7,
    ) -> None:
        self.q_grid = q_grid
        self.static_structure = static_structure
        self.diffusion_constants = diffusion_constants
        self.kernel_builder = kernel_builder
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        self.mixing = mixing

    def integrate(self, n_steps: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_q = len(self.q_grid)
        phi = np.zeros((n_steps, n_q, 2, 2), dtype=np.float64)
        phi_dot = np.zeros_like(phi)
        memory = np.zeros_like(phi)

        s_inv = np.zeros_like(self.static_structure)
        for q_idx in range(n_q):
            s_inv[q_idx] = np.linalg.inv(self.static_structure[q_idx])

        phi[0] = self.static_structure
        for q_idx in range(n_q):
            omega = self._omega_matrix(q_idx)
            phi_dot[0, q_idx] = -omega @ phi[0, q_idx]

        for step in range(1, n_steps):
            phi_new = np.copy(phi[step - 1]) + self.dt * phi_dot[step - 1]
            for iteration in range(self.max_iter):
                phi_snapshot = np.copy(phi_new)
                for q_idx in range(n_q):
                    memory_candidate = self.kernel_builder.evaluate(q_idx, phi_snapshot)
                    memory[step, q_idx] = memory_candidate
                phi_candidate = np.copy(phi_new)
                for q_idx in range(n_q):
                    omega = self._omega_matrix(q_idx)
                    convolution = self._time_convolution(q_idx, step, memory, phi_dot)
                    rhs = -omega @ phi_snapshot[q_idx] - convolution
                    phi_candidate[q_idx] = phi[step - 1, q_idx] + self.dt * rhs
                    phi_dot[step, q_idx] = rhs
                delta = np.max(np.abs(phi_candidate - phi_new))
                phi_new = self.mixing * phi_candidate + (1.0 - self.mixing) * phi_new
                if delta < self.tol:
                    break
            phi[step] = phi_new
        phi_norm = self._normalise(phi)
        return phi, phi_norm

    def _omega_matrix(self, q_idx: int) -> NDArray[np.float64]:
        q = self.q_grid[q_idx]
        d0 = self.diffusion_constants
        s_inv = np.linalg.inv(self.static_structure[q_idx])
        omega = np.diag(d0 * q * q) @ s_inv
        return omega

    def _time_convolution(
        self,
        q_idx: int,
        step: int,
        memory: NDArray[np.float64],
        phi_dot: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        acc = np.zeros((2, 2), dtype=np.float64)
        for m in range(step):
            weight = 0.5 if (m == 0 or m == step - 1) else 1.0
            acc += weight * memory[step - m, q_idx] @ phi_dot[m, q_idx]
        return self.dt * acc

    def _normalise(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        phi_norm = np.zeros_like(phi)
        for q_idx in range(len(self.q_grid)):
            s_inv = np.linalg.inv(self.static_structure[q_idx])
            for step in range(phi.shape[0]):
                phi_norm[step, q_idx] = phi[step, q_idx] @ s_inv
        return phi_norm


def build_diffusion_constants(species: Tuple[Species, Species], temperature: float) -> NDArray[np.float64]:
    k_b = 1.0
    diffusion = np.array([k_b * temperature / (s.mass) for s in species], dtype=np.float64)
    return diffusion


def run_simulation(
    q_min: float = 0.2,
    q_max: float = 20.0,
    n_q: int = 128,
    temperature: float = 1.5,
    dt: float = 1e-2,
    n_steps: int = 100,
    angular_method: str = "quadrature",
    n_theta: int = 256,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convenience function that runs a full MCT simulation."""

    q_grid = np.linspace(q_min, q_max, n_q)
    species = (
        Species("A", density=0.6, mass=1.0),
        Species("B", density=0.4, mass=1.5),
    )
    lj_params = LennardJonesParameters(
        epsilon={
            ("A", "A"): 1.0,
            ("B", "B"): 0.5,
            ("A", "B"): 0.8,
        },
        sigma={
            ("A", "A"): 1.0,
            ("B", "B"): 0.88,
            ("A", "B"): 0.94,
        },
        r_cutoff=4.5,
    )
    structure_builder = StaticStructureBuilder(species, lj_params, temperature)
    s_q, c_q = structure_builder.build(q_grid)
    integrator = AngularIntegrator(method=angular_method, n_points=n_theta)
    kernel_builder = MemoryKernelBuilder(q_grid, s_q, c_q, species, integrator)
    diffusion_constants = build_diffusion_constants(species, temperature)
    solver = PredictorCorrectorSolver(q_grid, s_q, diffusion_constants, kernel_builder, dt)
    phi, phi_norm = solver.integrate(n_steps)
    return phi, phi_norm, s_q


if __name__ == "__main__":
    phi, phi_norm, s_q = run_simulation(n_steps=20)
    print("Simulation completed.")
    print("Final normalised correlator at largest q:")
    print(phi_norm[-1, -1])
