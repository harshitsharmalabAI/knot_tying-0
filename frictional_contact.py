"""PyElastica rod self-contact with Coulomb friction."""
import numpy as np
from numba import njit
import elastica as ea
from elastica._contact_functions import _batch_product_k_ik_to_ik, _dot_product, _find_min_dist, _norm


@njit(cache=True)
def self_contact_with_friction(positions, radius, lengths, tangents, velocities, forces,
                               k, nu, slip_damping, friction):
    n = positions.shape[1]
    edges = _batch_product_k_ik_to_ik(lengths, tangents)

    for i in range(n):
        skip = int(1 + np.ceil(0.8 * np.pi * radius[i] / lengths[i]))   # neighbours never count
        for j in range(i - skip, -1, -1):
            radii_sum = radius[i] + radius[j]
            length_sum = lengths[i] + lengths[j]
            x_i = positions[..., i]
            x_j = positions[..., j]
            if _norm(x_i - x_j) >= (radii_sum + length_sum):
                continue

            normal, _, _ = _find_min_dist(x_i, edges[..., i], x_j, edges[..., j])
            distance = _norm(normal)
            normal /= distance
            overlap = radii_sum - distance
            if overlap < -1e-5:
                continue
            touching = (overlap > 0.0) * 1.0

            relative_velocity = 0.5 * ((velocities[..., i] + velocities[..., i + 1])
                                       - (velocities[..., j] + velocities[..., j + 1]))
            normal_force = 0.5 * touching * (nu * _dot_product(relative_velocity, normal) + k * overlap)
            force = normal_force * normal

            slip = relative_velocity - _dot_product(relative_velocity, normal) * normal
            slip_speed = np.linalg.norm(slip)
            friction_force = min(slip_damping * slip_speed, friction * abs(normal_force))
            force += friction_force * (slip / (slip_speed + 1e-14))

            if i == n - 1:
                forces[..., i] -= force * 4 / 3
                forces[..., i + 1] -= force * 2 / 3
            else:
                forces[..., i] -= force
                forces[..., i + 1] -= force
            if j == 0:
                forces[..., j] += force * 2 / 3
                forces[..., j + 1] += force * 4 / 3
            else:
                forces[..., j] += force
                forces[..., j + 1] += force


class FrictionalSelfContact(ea.RodSelfContact):
    """Self-contact with sliding force min(c * slip speed, mu * normal force)."""

    def __init__(self, k, nu, velocity_damping_coefficient, friction_coefficient):
        super().__init__(k=k, nu=nu)
        self.velocity_damping_coefficient = np.float64(velocity_damping_coefficient)
        self.friction_coefficient = np.float64(friction_coefficient)

    def apply_contact(self, system_one, system_two, time=np.float64(0.0)):
        self_contact_with_friction(
            system_one.position_collection[..., :-1], system_one.radius, system_one.lengths,
            system_one.tangents, system_one.velocity_collection, system_one.external_forces,
            self.k, self.nu, self.velocity_damping_coefficient, self.friction_coefficient)
