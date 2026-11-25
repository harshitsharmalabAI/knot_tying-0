"""Checks for wound_tissue.py, each against an answer worked out beforehand."""
import numpy as np
from wound_tissue import WoundFlaps, TissueDrag, wound_gap, add_wound

failures = 0


def check(ok, what):
    global failures
    failures += not ok
    print(("PASS " if ok else "FAIL ") + what)


class Thread:
    def __init__(self, pos, mass=None, rest=None):
        self.position_collection = pos.copy()
        self.velocity_collection = np.zeros_like(pos)
        self.external_forces = np.zeros_like(pos)
        self.mass, self.rest_lengths = mass, rest


cut, bite, k, contact_k = 0.004, 0.018, 20.0, 2e3
left, right = np.array([-(cut / 2 + bite), 0, 0]), np.array([cut / 2 + bite, 0, 0])
start = np.stack([left, [0, 0, 0.0], right], axis=1)
wound = dict(left_exit=0, right_exit=2, edge_bite=bite)

thread = Thread(start)
flaps = WoundFlaps(0, 2, left, right, k, 0.0, contact_k, bite)
flaps.apply_forces(thread)
check(np.allclose(thread.external_forces, 0) and np.isclose(wound_gap(start, wound), cut),
      "at rest: no force, gap equals the cut")

thread = Thread(start)
thread.position_collection[0, [0, 2]] += [1e-3, -1e-3]
flaps.apply_forces(thread)
check(np.isclose(thread.external_forces[0, 0], -k * 1e-3) and np.isclose(thread.external_forces[0, 2], k * 1e-3)
      and np.isclose(wound_gap(thread.position_collection, wound), cut - 2e-3),
      "exits pulled in 1 mm each: pushed back by k * 1 mm, gap 2 mm smaller")

thread = Thread(start)
thread.position_collection[0, [0, 2]] += [2.5e-3, -2.5e-3]
flaps.apply_forces(thread)
check(np.isclose(thread.external_forces[0, 0], -k * 2.5e-3 - contact_k * 1e-3),
      "edges overlapping by 1 mm: contact adds contact_k * 1 mm")

thread = Thread(start)
thread.velocity_collection[:, 1] = [0.1, -0.2, 0.3]
TissueDrag([1], 5.0).apply_forces(thread)
check(np.allclose(thread.external_forces[:, 1], [-0.5, 1.0, -1.5]) and np.allclose(thread.external_forces[:, [0, 2]], 0),
      "drag acts on the buried node only, against its velocity")

mass, dt = 0.1 / 176, 0.0025 * 0.889 / 175


def settle(pull, steps=400000):
    thread = Thread(start)
    flaps = WoundFlaps(0, 2, left, right, k, 2 * np.sqrt(k * mass), contact_k, bite)
    for _ in range(steps):
        thread.external_forces[:] = 0
        flaps.apply_forces(thread)
        d = thread.position_collection[:, 2] - thread.position_collection[:, 0]
        u = d / np.linalg.norm(d)
        thread.external_forces[:, 0] += pull * u
        thread.external_forces[:, 2] -= pull * u
        thread.velocity_collection[:, [0, 2]] += dt * thread.external_forces[:, [0, 2]] / mass
        thread.position_collection += dt * thread.velocity_collection
    return wound_gap(thread.position_collection, wound)


check(np.isclose(settle(k * cut / 4), cut / 2, rtol=1e-3), "a pull of k * cut / 4 closes the cut halfway")
expected = -(0.2 - k * cut / 2) / (contact_k + k / 2)
check(np.isclose(settle(0.2), expected, rtol=1e-3), f"a 0.2 N pull presses the edges to {expected*1e3:.3f} mm")


class Sim:
    def __init__(self):
        self.forcing = []

    def add_forcing_to(self, rod):
        sim = self

        class Using:
            def using(self, cls, *args):
                sim.forcing.append(cls(*args))
        return Using()


import tension_knot
pos, length, stitch = tension_knot.build_thread()
n = pos.shape[1] - 1
rod = Thread(pos, mass=np.full(n + 1, 0.1 * length / 0.889 / n), rest=np.full(n, length / n))
sim = Sim()
wound = add_wound(sim, rod, stitch, 0.0025 * length / n, 1e6, tension_knot.THREAD_RADIUS)
check(pos[0, wound["left_exit"]] < pos[0, wound["right_exit"]], "real layout: left exit is the one with smaller x")
check(np.isclose(wound_gap(pos, wound), 0.004), "real layout: gap starts at 4 mm")
rod.external_forces[:] = 0
sim.forcing[0].apply_forces(rod)
check(np.allclose(rod.external_forces, 0), "real layout: no force at the start")

print("\nALL PASS" if failures == 0 else f"\n{failures} FAILED")
