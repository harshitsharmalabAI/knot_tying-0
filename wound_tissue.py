"""Soft wound: tissue springs at the stitch exits, contact at the cut, drag on the buried stitch."""
import numpy as np
import elastica as ea


class WoundFlaps(ea.NoForces):
    def __init__(self, left_exit, right_exit, left_anchor, right_anchor,
                 stiffness, damping, contact_stiffness, edge_bite):
        self.left, self.right = int(left_exit), int(right_exit)
        self.left_anchor = np.array(left_anchor, dtype=float)
        self.right_anchor = np.array(right_anchor, dtype=float)
        self.k, self.c = float(stiffness), float(damping)
        self.contact_k, self.edge_bite = float(contact_stiffness), float(edge_bite)
        self.normal = np.array([1.0, 0.0, 0.0])          # across the cut

    def apply_forces(self, system, time=0.0):
        x, v, f = system.position_collection, system.velocity_collection, system.external_forces
        for i, anchor in ((self.left, self.left_anchor), (self.right, self.right_anchor)):
            f[:, i] += -self.k * (x[:, i] - anchor) - self.c * v[:, i]
        gap = np.dot(x[:, self.right] - x[:, self.left], self.normal) - 2.0 * self.edge_bite
        if gap < 0.0:
            push = self.contact_k * (-gap) * self.normal
            f[:, self.left] -= push
            f[:, self.right] += push

    def apply_torques(self, system, time=0.0):
        pass


class TissueDrag(ea.NoForces):
    def __init__(self, nodes, damping):
        self.nodes = np.asarray(nodes, dtype=int)
        self.c = float(damping)

    def apply_forces(self, system, time=0.0):
        system.external_forces[:, self.nodes] -= self.c * system.velocity_collection[:, self.nodes]

    def apply_torques(self, system, time=0.0):
        pass


def wound_gap(pos, wound):
    return float(pos[0, wound["right_exit"]] - pos[0, wound["left_exit"]] - 2.0 * wound["edge_bite"])


def add_wound(sim, rod, stitch, dt, E, radius, cut_width=0.004, pull_strain=0.40,
              closing_fraction=0.25, contact_stiffness=5e4, buried_damping=0.5):
    """Attach the wound flaps that hold the stitch."""
    nodes = np.sort(np.asarray(stitch, dtype=int))
    first, last = nodes[0], nodes[-1]
    x = rod.position_collection
    left, right = (first, last) if x[0, first] < x[0, last] else (last, first)
    span = x[0, right] - x[0, left]
    edge_bite = 0.5 * (span - cut_width)
    assert edge_bite > 0, "the stitch is narrower than the cut"
    node_mass = float(np.min(rod.mass[nodes]))
    pull_tension = E * np.pi * radius**2 * pull_strain
    stiffness = 2.0 * closing_fraction * pull_tension / cut_width
    damping = 2.0 * np.sqrt(stiffness * node_mass)
    assert max(damping, buried_damping) < 0.2 * (2.0 * node_mass / dt), "tissue damping too high for dt"
    for k in (stiffness, contact_stiffness):
        assert dt * np.sqrt(k / node_mass) < 0.2, "tissue spring too stiff for dt"
    buried = np.arange(first + 1, last)
    sim.add_forcing_to(rod).using(WoundFlaps, left, right, x[:, left].copy(), x[:, right].copy(),
                                  stiffness, damping, contact_stiffness, edge_bite)
    sim.add_forcing_to(rod).using(TissueDrag, buried, buried_damping)
    print(f"wound: cut {cut_width*1e3:.1f} mm, exits nodes {left}/{right} ({span*1e3:.1f} mm apart), "
          f"edge bite {edge_bite*1e3:.1f} mm, {len(buried)} buried nodes, "
          f"k {stiffness:.0f} N/m per flap (closes at ~{closing_fraction*pull_tension:.1f} N), c {damping:.2f}")
    return dict(left_exit=int(left), right_exit=int(right), edge_bite=edge_bite, cut_width=cut_width,
                stiffness=stiffness, damping=damping)


def draw_flaps(ax, pos, wound, depth=0.01, reach=0.05, color="0.8"):
    """Side view: the two flaps as grey blocks under the surface."""
    left, right, bite = pos[0, wound["left_exit"]], pos[0, wound["right_exit"]], wound["edge_bite"]
    ax.fill_between([left - reach, left + bite], -depth, 0.0, color=color, zorder=0)
    ax.fill_between([right - bite, right + reach], -depth, 0.0, color=color, zorder=0)
