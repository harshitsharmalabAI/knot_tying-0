"""tension_knot.py with extra options for the course experiments (identical with defaults)."""
import json
import os
import argparse
import csv
import time

import numpy as np
import elastica as ea
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from elastica.rod.knot_theory import compute_writhe

from frictional_contact import FrictionalSelfContact
import knot_check
from wound_tissue import add_wound, wound_gap, draw_flaps

THREAD_RADIUS = 0.005
TAIL_LEN = 0.15
STITCH_LEN = 0.04
LONG_LEN = 0.19
RESERVE_LEN = 0.45
ELEMENT_LEN = 0.00506
YOUNGS_MODULUS = 1e6
CUT_WIDTH = 0.004
WOUND_STIFFNESS = 0.25

B_RADIUS = 0.015
B_LENGTH = 0.25
B_HEIGHT = 0.12
B_TIP_X = 0.06
JAW_REACH = 0.025
ORBIT_RADIUS = 0.06
WRAP_X = [-0.015, -0.027, -0.039]
DRAW_BACK = np.array([-0.245, 0.0, 0.0])

FEED_TENSION = 2.0
FEED_SPEED = 0.15
DRAW_TENSION = 1.5
CINCH_TENSION = 10.0
SPEED_OUT_A_DRAW = 0.15
SPEED_OUT_A_CINCH = 0.10
SPEED_OUT_B = 0.05
SPEED_IN = 0.05
SLACK_TENSION = 0.5
A_TRAVEL_LIMITS = (-0.15, 0.70)
B_TRAVEL_LIMITS = (-0.10, 0.20)
GRIP_STIFFNESS = 2e4
GRIP_MAX_FORCE = 40.0
GRIP_SOFTENING = 1e-3
THREAD_FRICTION = 0.3
CLEARANCE = 0.03


def thread_path(reserve=0.0):
    """Dense starting path: tail -> stitch -> long strand rising in front of B -> reserve."""
    pts = [[0.02 + TAIL_LEN * (1 - s), 0.0, 0.0] for s in np.linspace(0, 1, 800)]
    pts += [[0.02 - STITCH_LEN * s, 0.0, 0.0] for s in np.linspace(0, 1, 200)[1:]]
    r = 0.02
    for s in np.linspace(0, 1, 200)[1:]:
        a = np.pi / 2 * s
        pts.append([-0.02 - r * np.sin(a), 0.0, r - r * np.cos(a)])
    pts += [[-0.04, 0.0, 0.02 + 0.02 * s] for s in np.linspace(0, 1, 100)[1:]]
    for s in np.linspace(0, 1, 200)[1:]:
        b = np.pi / 2 * s
        pts.append([-0.04, r - r * np.cos(b), 0.04 + r * np.sin(b)])
    rise = LONG_LEN - 0.02 - np.pi * r
    pts += [[-0.04, r + rise * s, 0.06] for s in np.linspace(0, 1, 1200)[1:]]
    if reserve > 0:
        top = np.array(pts[-1])
        pts += [top + np.array([reserve * s, 0.0, 0.0]) for s in np.linspace(0, 1, 600)[1:]]
    path = np.array(pts).T
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=1), axis=0))])
    return path, arc


WORKING_LEN = thread_path()[1][-1]
N_WORKING = int(round(WORKING_LEN / ELEMENT_LEN))
DL = WORKING_LEN / N_WORKING
N_RESERVE = int(round(RESERVE_LEN / DL))
N_ELEMENTS = N_WORKING + N_RESERVE
GRIP_NODE = N_WORKING
RESERVE_DIR = np.array([1.0, 0.0, 0.0])


def build_thread():
    path, arc = thread_path(N_RESERVE * DL)
    s = np.linspace(0, WORKING_LEN + N_RESERVE * DL, N_ELEMENTS + 1)
    pos = np.array([np.interp(s, arc, path[k]) for k in range(3)])
    stitch = np.where((s >= TAIL_LEN - 1e-9) & (s <= TAIL_LEN + STITCH_LEN + 1e-9))[0]
    return pos, s[-1], stitch


def add_thread_noise(pos, stitch, amplitude, seed):
    """Smooth sideways offset of the tail and long strand, zero at the stitch and at A's grip."""
    rng = np.random.default_rng(seed)
    pos = pos.copy()
    tail = np.arange(stitch[0], -1, -1)
    s = np.linspace(0.0, 1.0, len(tail))
    shape = sum(rng.normal() * np.sin((k - 0.5) * np.pi * s) for k in range(1, 4))
    pos[2, tail] += amplitude * shape / max(np.abs(shape).max(), 1e-12)
    strand = np.arange(stitch[-1], GRIP_NODE + 1)        # from the stitch to A's grip
    s = np.linspace(0.0, 1.0, len(strand))
    shape = sum(rng.normal() * np.sin(k * np.pi * s) for k in range(1, 4))
    pos[2, strand] += amplitude * shape / max(np.abs(shape).max(), 1e-12)
    return pos


def frames_along(pos):
    tangents = np.diff(pos, axis=1)
    tangents /= np.linalg.norm(tangents, axis=0)
    D = np.zeros((3, 3, tangents.shape[1]))
    d1 = np.cross(tangents[:, 0], [0.0, 0.0, 1.0])
    d1 /= np.linalg.norm(d1)
    for i in range(tangents.shape[1]):
        t = tangents[:, i]
        d1 = d1 - (d1 @ t) * t
        d1 /= np.linalg.norm(d1)
        D[0, :, i], D[1, :, i], D[2, :, i] = d1, np.cross(t, d1), t
    return D


B_START = np.array([B_TIP_X, B_HEIGHT, 0.0])
TAIL_END = np.array([0.02 + TAIL_LEN, 0.0, 0.0])
B_AT_TAIL = np.array([TAIL_END[0] - JAW_REACH, THREAD_RADIUS + B_RADIUS + 0.005, 0.0])
B_SHIFT = B_AT_TAIL - B_START
B_AXIS = np.array([1.0, 0.0, 0.0])        # B's tip points +x throughout


def orbit(x, angle):
    """A point circling B's axis; angle 0 is in front (+z)."""
    return np.array([x, B_HEIGHT + ORBIT_RADIUS * np.sin(angle), ORBIT_RADIUS * np.cos(angle)])


TIME_SCALE = 1.0
RELEASE = False


def stages(a_start):
    """(duration, A target, B tip target, B gripping?, label) for each stage."""
    return [(d * TIME_SCALE, a, b, g, label) for d, a, b, g, label in _stages(a_start)]


def _stages(a_start):
    wrapped = orbit(WRAP_X[2], 4 * np.pi)
    return [
        (0.5, a_start, B_START, False, "settle"),
        (2.0, orbit(WRAP_X[0], 0.0), B_START, False, "A brings the long strand beside B"),
        (3.0, lambda f: orbit(WRAP_X[0] + (WRAP_X[1] - WRAP_X[0]) * wrap_ease(f), 2 * np.pi * wrap_ease(f)),
         B_START, False, "wrap 1: A carries it round B"),
        (3.0, lambda f: orbit(WRAP_X[1] + (WRAP_X[2] - WRAP_X[1]) * wrap_ease(f), 2 * np.pi * (1 + wrap_ease(f))),
         B_START, False, "wrap 2: round again - surgeon's double throw"),
        (3.0, wrapped + B_SHIFT, B_AT_TAIL, False, "B travels to the tail, level; A follows"),
        (0.5, wrapped + B_SHIFT, B_AT_TAIL, True, "B grips the tail"),
        (3.0, wrapped + B_SHIFT, B_AT_TAIL + DRAW_BACK, True, "B draws the tail back through the wraps"),
        (4.0, wrapped + B_SHIFT, B_AT_TAIL + DRAW_BACK, True, "cinch: both pull apart to T"),
        (3.0, wrapped + B_SHIFT, B_AT_TAIL + DRAW_BACK, True, "hold the knot under tension"),
    ] + ([(3.0, wrapped + B_SHIFT, B_AT_TAIL + DRAW_BACK, False, "release: both instruments let go")]
         if RELEASE else [])


def ease(f):
    f = min(1.0, max(0.0, f))
    return 0.5 - 0.5 * np.cos(np.pi * f)


PROFILE = "cosine"
EASE_WRAPS = False


def move_ease(f):
    """Time scaling s(f) of a straight move, 0 -> 1."""
    f = min(1.0, max(0.0, f))
    if PROFILE == "linear":
        return f
    if PROFILE == "cubic":
        return 3 * f ** 2 - 2 * f ** 3
    if PROFILE == "quintic":
        return 10 * f ** 3 - 15 * f ** 4 + 6 * f ** 5
    return 0.5 - 0.5 * np.cos(np.pi * f)


def wrap_ease(f):
    return move_ease(f) if EASE_WRAPS else f


def servo_step(travel, tension, target, speed_out, step):
    """Pull out while 20% under target, move in while 20% over."""
    if tension < 0.8 * target:
        return travel + speed_out * step
    if tension > 1.2 * target:
        return travel - SPEED_IN * step
    return travel


def feedback_speed(kind, kp, ki, integral, error, step, speed_out, speed_in=None):
    """Servo pull-out speed from the tension error (P or PI with anti-windup)."""
    speed_in = SPEED_IN if speed_in is None else speed_in
    speed = kp * error
    if kind == "pi":
        trial = speed + ki * (integral + error * step)
        if -speed_in < trial < speed_out:
            integral += error * step
        speed += ki * integral
    return float(np.clip(speed, -speed_in, speed_out)), integral


class Plan:
    """Where A and B's tip should be at time t, plus the servos' offsets."""

    def __init__(self, stage_list, a_start):
        self.stages = stage_list
        self.ends = np.cumsum([s[0] for s in stage_list])
        self.a_start = a_start
        self.a_end = []
        prev = a_start
        for _, a, _, _, _ in stage_list:
            prev = a(1.0) if callable(a) else np.array(a)
            self.a_end.append(prev)
        labels = [s[4] for s in stage_list]
        self.travel_stage = labels.index("B travels to the tail, level; A follows")
        self.draw_stage = labels.index("B draws the tail back through the wraps")
        self.cinch_stage = labels.index("cinch: both pull apart to T")
        self.hold_stage = labels.index("hold the knot under tension")
        self.a_offset = np.zeros(3)
        self.b_offset = np.zeros(3)

    def stage_at(self, t):
        for k, end in enumerate(self.ends):
            if t <= end:
                return k, (t - (end - self.stages[k][0])) / self.stages[k][0]
        return len(self.stages) - 1, 1.0

    def a_pos(self, t):
        k, f = self.stage_at(t)
        target = self.stages[k][1]
        if callable(target):
            return target(f)
        prev = self.a_start if k == 0 else self.a_end[k - 1]
        pos = prev + (np.array(target) - prev) * move_ease(f)
        return pos + self.a_offset if k >= self.travel_stage else pos

    def b_tip(self, t):
        k, f = self.stage_at(t)
        prev = B_START if k == 0 else np.array(self.stages[k - 1][2])
        pos = prev + (np.array(self.stages[k][2]) - prev) * move_ease(f)
        return pos + self.b_offset if k >= self.cinch_stage else pos

    def gripping(self, t):
        return self.stages[self.stage_at(t)[0]][3]


def run(argv=None, live_hook=None, live_dt=0.005):
    """Run the simulation; live_hook(state) is called every live_dt if given."""
    global PROFILE, EASE_WRAPS, ORBIT_RADIUS, WRAP_X, TIME_SCALE, RELEASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", metavar="NPZ", help="start from a saved checkpoint")
    parser.add_argument("--t-cinch", type=float, default=CINCH_TENSION)
    parser.add_argument("--t-draw", type=float, default=DRAW_TENSION)
    parser.add_argument("--mu-thread", type=float, default=THREAD_FRICTION)
    parser.add_argument("--grip-fmax", type=float, default=GRIP_MAX_FORCE)
    parser.add_argument("--stop-after", type=int, default=None, metavar="STAGE")
    parser.add_argument("--diagnose", action="store_true", help="print details every 0.1 s")
    parser.add_argument("--controller", choices=["deadband", "p", "pi"], default="deadband",
                        help="tension control in the cinch and hold stages")
    parser.add_argument("--kp", type=float, default=0.01, help="P gain, m/s per N")
    parser.add_argument("--ki", type=float, default=0.01, help="I gain, m/s per N s")
    parser.add_argument("--profile", choices=["cosine", "linear", "cubic", "quintic"], default="cosine")
    parser.add_argument("--ease-wraps", action="store_true")
    parser.add_argument("--orbit-radius", type=float, default=ORBIT_RADIUS)
    parser.add_argument("--wrap-shift", type=float, default=0.0, help="move all wraps along B (+ = toward the tip), m")
    parser.add_argument("--thread-noise", type=float, default=0.0, help="max sideways offset of the start shape, m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--release", action="store_true", help="after the hold, both instruments let go (release test)")
    parser.add_argument("--time-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    PROFILE, EASE_WRAPS, ORBIT_RADIUS, TIME_SCALE = args.profile, args.ease_wraps, args.orbit_radius, args.time_scale
    RELEASE = args.release
    WRAP_X = [x + args.wrap_shift for x in WRAP_X]
    cinch_tension, draw_tension = args.t_cinch, args.t_draw
    grip_max_force = args.grip_fmax
    verbose = args.diagnose or args.resume is not None

    pos, length, stitch = build_thread()
    if args.thread_noise > 0:
        pos = add_thread_noise(pos, stitch, args.thread_noise, args.seed)
    a_start = pos[:, GRIP_NODE].copy()
    plan = Plan(stages(a_start), a_start)
    print(f"thread {length:.3f} m, {N_ELEMENTS} elements, stitch = nodes "
          f"{stitch[0]}-{stitch[-1]} across a {CUT_WIDTH*1000:.0f} mm cut")
    print(f"one throw, then cinch to {cinch_tension} N and hold; reserve {RESERVE_LEN*100:.0f} cm; "
          f"thread mu {args.mu_thread}; grip capped at {grip_max_force:.0f} N")

    class Sim(ea.BaseSystemCollection, ea.Constraints, ea.Contact, ea.Forcing, ea.Damping):
        pass

    sim = Sim()
    area = np.pi * THREAD_RADIUS ** 2
    thread_mass = 0.1 * length / 0.889
    density = thread_mass / (area * length)
    E = YOUNGS_MODULUS
    dt = 0.0025 * length / N_ELEMENTS

    rod = ea.CosseratRod.straight_rod(
        N_ELEMENTS, np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
        length, THREAD_RADIUS, density, youngs_modulus=E, shear_modulus=E / 1.5)
    rod.position_collection[:] = pos
    rod.director_collection[:] = frames_along(pos)
    sim.append(rod)
    sim.dampen(rod).using(ea.AnalyticalLinearDamper, damping_constant=2.0 * 0.889, time_step=dt)
    if hasattr(ea, "LaplaceDissipationFilter") and not args.no_filter:
        sim.dampen(rod).using(ea.LaplaceDissipationFilter, filter_order=3)
    sim.detect_contact_between(rod, rod).using(
        FrictionalSelfContact, k=1e4, nu=10, velocity_damping_coefficient=1e2,
        friction_coefficient=args.mu_thread)
    sim.add_forcing_to(rod).using(ea.GravityForces, acc_gravity=np.array([0.0, -9.81, 0.0]))
    tissue = ea.Plane(plane_origin=np.array([0.0, -THREAD_RADIUS, 0.0]),
                      plane_normal=np.array([0.0, 1.0, 0.0]))
    sim.append(tissue)
    sim.detect_contact_between(rod, tissue).using(
        ea.RodPlaneContactWithAnisotropicFriction, k=1e3, nu=1e-1, slip_velocity_tol=1e-6,
        static_mu_array=np.array([0.6, 0.6, 0.6]), kinetic_mu_array=np.array([0.4, 0.4, 0.4]))
    wound = add_wound(sim, rod, stitch, dt, E=E, radius=THREAD_RADIUS, cut_width=CUT_WIDTH,
                      closing_fraction=WOUND_STIFFNESS)

    b_rod = ea.Cylinder(start=B_START - np.array([B_LENGTH, 0, 0]), direction=np.array([1.0, 0, 0]),
                        normal=np.array([0, 1.0, 0]), base_length=B_LENGTH,
                        base_radius=B_RADIUS, density=1000.0)
    sim.append(b_rod)
    sim.detect_contact_between(rod, b_rod).using(
        ea.RodCylinderContact, k=5e4, nu=10, velocity_damping_coefficient=1e2,
        friction_coefficient=0.2)

    hand = {"node": float(GRIP_NODE)}      # fractional while feeding
    servo = {"a": 0.0, "b": 0.0}
    feed_stages = {"settle", "A brings the long strand beside B", "wrap 1: A carries it round B",
                   "wrap 2: round again - surgeon's double throw"}
    target = None if draw_tension <= 0 else draw_tension
    servo_targets = {
        plan.travel_stage: ((target, SPEED_OUT_A_DRAW), None),
        plan.travel_stage + 1: ((target, SPEED_OUT_A_DRAW), None),
        plan.draw_stage: ((target, SPEED_OUT_A_DRAW), None),
        plan.cinch_stage: ((cinch_tension, SPEED_OUT_A_CINCH), (cinch_tension, SPEED_OUT_B)),
        plan.hold_stage: ((cinch_tension, SPEED_OUT_A_CINCH), (cinch_tension, SPEED_OUT_B)),
    }
    a_pull, b_pull = np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])
    control = {"stages": {plan.cinch_stage, plan.hold_stage}, "integral": {"a": 0.0, "b": 0.0}}

    def control_step(name, travel, tension_now, target_t, speed_out, step):
        if args.controller == "deadband":
            return servo_step(travel, tension_now, target_t, speed_out, step)
        speed, control["integral"][name] = feedback_speed(
            args.controller, args.kp, args.ki, control["integral"][name], target_t - tension_now, step, speed_out)
        return travel + speed * step

    grip_state = {"force": 0.0, "slip": 0.0, "capped": False}

    def tension(strain):
        return E * area * strain / (1.0 + strain)

    def strand_tensions(x):
        rest = rod.rest_lengths
        g = int(np.floor(hand["node"]))
        a_strain = np.linalg.norm(np.diff(x[:, g - 3:g + 1], axis=1), axis=0) / rest[g - 3:g] - 1.0
        tail_strain = np.linalg.norm(np.diff(x[:, :4], axis=1), axis=0) / rest[:3] - 1.0
        return float(tension(a_strain).mean()), float(tension(tail_strain).mean())

    def stitch_tension(x):
        seg = np.linalg.norm(np.diff(x[:, stitch[0]:stitch[-1] + 1], axis=1), axis=0)
        return float(tension(seg / rod.rest_lengths[stitch[0]:stitch[-1]] - 1.0).mean())

    class DriveThread(ea.ConstraintBase):
        """A holds the strand at hand['node'] with the reserve in a line behind it."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.last_t = None
            self.bend_original = None
            self.soft_from = None

        def released(self, t):
            return plan.stages[plan.stage_at(t)[0]][4].startswith("release")

        def constrain_values(self, system, time):
            t = time
            k = plan.stage_at(t)[0]
            if self.released(t):
                self.last_t = t
                if self.bend_original is not None and self.soft_from != "released":
                    system.bend_matrix[...] = self.bend_original
                    self.soft_from = "released"
                return
            step = t - self.last_t if self.last_t is not None else 0.0
            if 0.0 < step <= 10.0 * dt:
                if k in servo_targets:
                    self.run_servos(system, step, k)
                if plan.stages[k][4] in feed_stages:
                    if strand_tensions(system.position_collection)[0] > FEED_TENSION:
                        hand["node"] = min(hand["node"] + FEED_SPEED * step / DL, N_ELEMENTS - 3.0)
            self.last_t = t
            self.soften_grip(system)
            held = self.held_nodes()
            system.position_collection[:, held] = (
                plan.a_pos(t)[:, None] + np.outer(RESERVE_DIR, (held - hand["node"]) * DL))

        def held_nodes(self):
            return np.arange(int(np.ceil(hand["node"] - 1e-9)), N_ELEMENTS + 1)

        def soften_grip(self, system):
            first = int(np.ceil(hand["node"] - 1e-9)) - 1
            if first == self.soft_from:
                return
            bend = system.bend_matrix
            if self.bend_original is None:
                self.bend_original = bend.copy()
            bend[...] = self.bend_original
            bend[..., max(first, 0):] = self.bend_original[..., max(first, 0):] * GRIP_SOFTENING
            self.soft_from = first

        def run_servos(self, system, step, k):
            a_tension, tail_tension = strand_tensions(system.position_collection)
            a_target, b_target = servo_targets[k]
            rule = control_step if k in control["stages"] else (lambda _n, *a: servo_step(*a))
            if a_target is not None and a_target[0] is not None:
                servo["a"] = float(np.clip(rule("a", servo["a"], a_tension, a_target[0], a_target[1], step),
                                           *A_TRAVEL_LIMITS))
            if b_target is not None:
                servo["b"] = float(np.clip(rule("b", servo["b"], tail_tension, b_target[0], b_target[1], step),
                                           *B_TRAVEL_LIMITS))
            plan.a_offset = servo["a"] * a_pull
            plan.b_offset = servo["b"] * b_pull

        def constrain_rates(self, system, time):
            if self.released(time):
                return
            system.velocity_collection[:, self.held_nodes()] = 0.0

    class JawGrip(ea.NoForces):
        """B's jaws: a capped spring pulling the tail end to a point past B's tip."""

        def __init__(self, k, c, max_force):
            self.k, self.c, self.max_force = k, c, max_force
            self.offset, self.t_start = None, None

        def apply_forces(self, system, time=0.0):
            t = time
            if not plan.gripping(t):
                self.offset = None
                grip_state.update(force=0.0, slip=0.0, capped=False)
                return
            jaw = plan.b_tip(t) + JAW_REACH * B_AXIS
            end = system.position_collection[:, 0]
            if self.offset is None:
                self.offset, self.t_start = end - jaw, t
            target_pos = jaw + self.offset * (1.0 - ease((t - self.t_start) / 0.4))
            jaw_velocity = ((plan.b_tip(t + dt) + JAW_REACH * B_AXIS) - (plan.b_tip(t) + JAW_REACH * B_AXIS)) / dt
            force = self.k * (target_pos - end) - self.c * (system.velocity_collection[:, 0] - jaw_velocity)
            size = float(np.linalg.norm(force))
            capped = size > self.max_force
            if capped:
                force = force * (self.max_force / size)
            system.external_forces[:, 0] += force
            grip_state.update(force=min(size, self.max_force),
                              slip=float(np.linalg.norm(target_pos - end)), capped=capped)

        def apply_torques(self, system, time=0.0):
            pass

    end_mass = float(rod.mass[0])
    grip_damping = 2.0 * np.sqrt(GRIP_STIFFNESS * end_mass)
    assert dt * np.sqrt(GRIP_STIFFNESS / end_mass) < 0.2 and grip_damping < 0.4 * end_mass / dt
    assert grip_max_force < 0.9 * E * area
    sim.add_forcing_to(rod).using(JawGrip, GRIP_STIFFNESS, grip_damping, grip_max_force)

    class DriveB(ea.ConstraintBase):
        def __init__(self, directors, **kw):
            super().__init__(**kw)
            self.directors = directors.copy()

        def centre(self, t):
            return plan.b_tip(t) - (B_LENGTH / 2) * B_AXIS

        def constrain_values(self, system, time):
            system.position_collection[:, 0] = self.centre(time)
            system.director_collection[..., 0] = self.directors

        def constrain_rates(self, system, time):
            system.velocity_collection[:, 0] = (self.centre(time + dt) - self.centre(time)) / dt
            system.omega_collection[:, 0] = 0.0

    sim.constrain(rod).using(DriveThread)
    sim.constrain(b_rod).using(DriveB, directors=b_rod.director_collection[..., 0])
    sim.finalize()

    def clearance(t):
        """Gap between A's gripper and B's shaft or jaws."""
        a = plan.a_pos(t)
        tip = plan.b_tip(t)
        back = tip - B_LENGTH * B_AXIS
        shaft = tip - back
        along = np.clip(np.dot(a - back, shaft) / np.dot(shaft, shaft), 0.0, 1.0)
        to_shaft = np.linalg.norm(a - (back + along * shaft)) - B_RADIUS
        return float(min(to_shaft, np.linalg.norm(a - (tip + JAW_REACH * B_AXIS))))

    closest_planned = min(clearance(t) for t in np.arange(0.0, plan.ends[-1], 0.01))
    print(f"planned paths: A and B never closer than {closest_planned*100:.1f} cm")
    if closest_planned < CLEARANCE:
        raise SystemExit("planned paths bring A too close to B")

    def gap_to_b(p, t, gripping):
        tip = plan.b_tip(t)
        q = p[:, 4:] if gripping else p
        rel = q - tip[:, None]
        along = -(B_AXIS @ rel)
        inside = (along > 0) & (along < B_LENGTH)
        if not inside.any():
            return float("nan")
        radial = np.linalg.norm(rel - np.outer(B_AXIS, B_AXIS @ rel), axis=0)
        return float(radial[inside].min() - B_RADIUS - THREAD_RADIUS)

    def wraps_on_b(p, t):
        rel = p[:, stitch[-1] + 1:] - plan.b_tip(t)[:, None]
        along = -(B_AXIS @ rel)
        radial = np.linalg.norm(rel - np.outer(B_AXIS, B_AXIS @ rel), axis=0)
        return int(np.sum((along > -0.003) & (along < B_LENGTH) & (radial < B_RADIUS + THREAD_RADIUS + 0.003)))

    def throw_text(p):
        tail, long_ = p[:, :stitch[0] + 1], p[:, stitch[-1]:]
        d = np.linalg.norm(tail[:, :, None] - long_[:, None, :], axis=0)
        i, j = np.where(d < 3.0 * THREAD_RADIUS)
        if len(i) == 0:
            return "   strands not touching   "
        mid = 0.5 * (tail[:, i] + long_[:, j])
        centre = p[:, stitch].mean(axis=1)
        return f"   {np.abs(mid[0] - centre[0]).mean()*100:5.1f} cm, {(mid[1].mean() + THREAD_RADIUS)*100:5.1f} cm    "

    def winding(p):
        """Gauss linking integral between the tail and the long strand."""
        tail, long_ = p[:, :stitch[0] + 1], p[:, stitch[-1]:]
        m1, d1 = 0.5 * (tail[:, 1:] + tail[:, :-1]), np.diff(tail, axis=1)
        m2, d2 = 0.5 * (long_[:, 1:] + long_[:, :-1]), np.diff(long_, axis=1)
        r = m1[:, :, None] - m2[:, None, :]
        cross = np.cross(d1.T[:, None, :], d2.T[None, :, :])
        return float((np.einsum('kij,ijk->ij', r, cross) / (np.linalg.norm(r, axis=0) ** 3 + 1e-12)).sum() / (4 * np.pi))

    def part(i):
        return "tail" if i < stitch[0] else "stitch" if i <= stitch[-1] else "long strand"

    def inspect_state(x, v):
        strain = np.linalg.norm(np.diff(x, axis=1), axis=0) / rod.rest_lengths - 1.0
        speed = np.linalg.norm(v, axis=0)
        d = np.linalg.norm(x[:, :, None] - x[:, None, :], axis=0)
        idx = np.arange(x.shape[1])
        d = np.where(np.abs(idx[:, None] - idx[None, :]) > 4, d, np.inf)
        a, b = np.unravel_index(np.argmin(d), d.shape)
        return strain, speed, (a, b, d[a, b] - 2 * THREAD_RADIUS)

    def detail_line(t, x, v):
        strain, speed, (a, b, gap) = inspect_state(x, v)
        e, i = int(np.argmax(strain)), int(np.argmax(speed))
        a_tension, tail_tension = strand_tensions(x)
        return (f"    t {t:6.2f}s  strain max {strain.max()*100:5.1f}% (el {e:3d} {part(e):11s}, "
                f"{tension(strain[e]):4.1f} N)  strands A {a_tension:4.1f} N, tail {tail_tension:4.1f} N  "
                f"grip {grip_state['force']:4.1f} N{' CAPPED' if grip_state['capped'] else ''} "
                f"slip {grip_state['slip']*1e3:4.1f} mm  fastest {speed.max():5.2f} m/s "
                f"(node {i:3d} {part(i):11s})  closest {gap*1e3:+5.1f} mm ({part(a)}/{part(b)})  "
                f"wound {wound_gap(x, wound)*1e3:+4.1f} mm")

    def report(t, x, v):
        strain, speed, (a, b, gap) = inspect_state(x, v)
        print("   fastest nodes:  " + ",  ".join(f"{i} ({part(i)}) {speed[i]:.2f} m/s"
                                              for i in np.argsort(speed)[::-1][:4]))
        print("   most stretched: " + ",  ".join(f"el {e} ({part(e)}) {strain[e]*100:+.0f}%"
                                              for e in np.argsort(strain)[::-1][:4]))
        print(f"   closest self-approach {gap*1e3:+.1f} mm between nodes {a} ({part(a)}) and "
              f"{b} ({part(b)}); wound {wound_gap(x, wound)*1e3:+.1f} mm")

    stepper = ea.PositionVerlet()
    t, wall_start, snapshots = 0.0, time.time(), []
    first_stage = 0
    if args.resume:
        saved = np.load(args.resume)
        if saved["position"].shape[1] - 1 != N_ELEMENTS:
            raise SystemExit("checkpoint is from a different thread layout")
        rod.position_collection[:] = saved["position"]
        rod.velocity_collection[:] = saved["velocity"]
        rod.director_collection[:] = saved["director"]
        rod.omega_collection[:] = saved["omega"]
        hand["node"] = float(saved["hand_g"])
        servo["a"], servo["b"] = float(saved["s_a"]), float(saved["s_b"])
        plan.a_offset, plan.b_offset = servo["a"] * a_pull, servo["b"] * b_pull
        t = float(saved["t"])
        first_stage = int(saved["stage"]) + 1
        b_rod.position_collection[:, 0] = plan.b_tip(t) - (B_LENGTH / 2) * B_AXIS
        print(f"resumed from {args.resume} at t = {t:.3f} s; next stage {first_stage}: "
              f"{plan.stages[first_stage][4]}")

    max_speed = 2.0 * float(np.sqrt(E / density))
    max_strain = 3.0
    check_every, detail_every, frame_every = 0.005, 0.1, 0.05
    frames = {key: [] for key in ("t", "stage", "pos", "b_tip", "b_u", "a_pos", "hand_g", "grip", "wound")}
    next_frame = next_live = t
    labels = [s[4] for s in plan.stages]

    def save_frames():
        if frames["t"]:
            np.savez_compressed(
                "tension_knot_frames.npz", **{key: np.array(v) for key, v in frames.items()},
                labels=np.array(labels), bite=stitch, n=N_ELEMENTS, n_base=N_WORKING,
                b_len=B_LENGTH, b_r=B_RADIUS, r_rod=THREAD_RADIUS, jaw=JAW_REACH,
                wound_iL=wound["left_exit"], wound_iR=wound["right_exit"], edge_bite=wound["edge_bite"],
                frame_dt=frame_every)

    unstable = None
    closest = {"stage": np.inf, "run": np.inf}
    log_rows, stage_rows = [], []
    print(" stage                                          writhe  gap to B  winding  contour  "
          "throw: from stitch, height   wound   on B  grip   A str  stitch   A-B   wall")
    for k, end in enumerate(plan.ends):
        if k < first_stage:
            continue
        closest["stage"] = np.inf
        next_check, next_detail = t + check_every, t
        last_good = (t, rod.position_collection.copy(), rod.velocity_collection.copy(),
                     rod.director_collection.copy(), rod.omega_collection.copy())
        while t < end - 1e-12:
            t = stepper.step(sim, t, dt)
            if t < next_check:
                continue
            next_check += check_every
            x, v = rod.position_collection, rod.velocity_collection
            finite = np.all(np.isfinite(x)) and np.all(np.isfinite(v))
            if not finite or np.linalg.norm(v, axis=0).max() > max_speed or \
                    (np.linalg.norm(np.diff(x, axis=1), axis=0) / rod.rest_lengths - 1.0).max() > max_strain:
                unstable = (k, t, finite)
                break
            last_good = (t, x.copy(), v.copy(), rod.director_collection.copy(), rod.omega_collection.copy())
            gap = clearance(t)
            a_t, tail_t = strand_tensions(x)
            log_rows.append((round(t, 5), k, round(a_t, 4), round(tail_t, 4), round(servo["a"], 5),
                             round(servo["b"], 5), round(grip_state["force"], 3), round(gap, 5)))
            if gap < closest["stage"]:
                if gap < CLEARANCE <= closest["stage"]:
                    print(f"    WARNING t {t:.2f}s: A within {gap*100:.1f} cm of B", flush=True)
                closest["stage"] = gap
            if t >= next_frame:
                frames["t"].append(t); frames["stage"].append(k)
                frames["pos"].append(x.astype(np.float32))
                frames["b_tip"].append(plan.b_tip(t)); frames["b_u"].append(B_AXIS.copy())
                frames["a_pos"].append(plan.a_pos(t)); frames["hand_g"].append(hand["node"])
                frames["grip"].append(bool(plan.stages[k][3])); frames["wound"].append(wound_gap(x, wound))
                next_frame += frame_every
            if live_hook is not None and t >= next_live:
                next_live += live_dt
                a_tension, tail_tension = strand_tensions(x)
                live_hook(dict(
                    t=t, stage=k, label=labels[k], thread=x.copy(), b_tip=plan.b_tip(t), b_axis=B_AXIS.copy(),
                    a_pos=plan.a_pos(t), gripping=bool(plan.stages[k][3]), hand_node=hand["node"],
                    wound_gap=wound_gap(x, wound), a_tension=a_tension, tail_tension=tail_tension,
                    stitch_tension=stitch_tension(x), clearance=gap, stitch=stitch, n_elements=N_ELEMENTS,
                    thread_radius=THREAD_RADIUS, b_length=B_LENGTH, b_radius=B_RADIUS,
                    left_exit=wound["left_exit"], right_exit=wound["right_exit"], edge_bite=wound["edge_bite"],
                    wall=time.time() - wall_start))
            if verbose and t >= next_detail:
                print(detail_line(t, x, v), flush=True)
                next_detail += detail_every
        save_frames()
        closest["run"] = min(closest["run"], closest["stage"])
        if unstable is not None:
            bad_stage, bad_t, finite = unstable
            print(f"\nUNSTABLE during stage {bad_stage}: {labels[bad_stage]}, t = {bad_t:.3f} s")
            if finite:
                print("   at the moment it was caught:")
                report(bad_t, rod.position_collection, rod.velocity_collection)
            good_t, good_x, good_v, good_d, good_w = last_good
            print(f"   last healthy state, t = {good_t:.3f} s:")
            report(good_t, good_x, good_v)
            np.savez("unstable_last_good.npz", t=good_t, position=good_x, velocity=good_v,
                     director=good_d, omega=good_w, stage=bad_stage)
            snapshots.append((good_x.copy(), plan.b_tip(good_t), f"LAST HEALTHY, t={good_t:.2f}s",
                              plan.a_pos(good_t), hand["node"]))
            break

        p = rod.position_collection.copy()
        snapshots.append((p, plan.b_tip(t), labels[k], plan.a_pos(t), hand["node"]))
        checkpoint = {"B grips the tail": "tension_checkpoint_before_draw.npz",
                      "B draws the tail back through the wraps": "tension_checkpoint_after_throw.npz"}.get(labels[k])
        if checkpoint:
            np.savez(checkpoint, t=t, stage=k, position=rod.position_collection,
                     velocity=rod.velocity_collection, director=rod.director_collection,
                     omega=rod.omega_collection, s_a=servo["a"], s_b=servo["b"],
                     hand_g=hand["node"], n=N_ELEMENTS)
            print(f"     (state saved to {checkpoint})")
        gap = gap_to_b(p, t, plan.stages[k][3])
        gap_text = "   -   " if np.isnan(gap) else f"{gap*1000:+6.1f}mm"
        a_tension, tail_tension = strand_tensions(p)
        writhe = float(compute_writhe(p[None], 1.0, "next_tangent")[0])
        print(f" {k}: {labels[k]:45s} {writhe:+6.2f}  {gap_text}  {winding(p):+6.2f}  "
              f"{np.linalg.norm(np.diff(p, axis=1), axis=0).sum():6.3f}m  {throw_text(p)}  "
              f"{wound_gap(p, wound)*1000:+5.1f}mm  {wraps_on_b(p, t):4d}  {grip_state['force']:4.1f}N  "
              f"{a_tension:4.1f}N  {stitch_tension(p):5.1f}N  "
              f"{closest['stage']*100 if np.isfinite(closest['stage']) else 99:4.1f}cm  "
              f"{time.time() - wall_start:5.0f}s", flush=True)
        stage_rows.append(dict(stage=k, label=labels[k], t=round(t, 4), writhe=round(writhe, 4),
                               winding=round(winding(p), 4), a_tension=round(a_tension, 3),
                               tail_tension=round(tail_tension, 3), wraps_on_b=wraps_on_b(p, t),
                               contour=round(float(np.linalg.norm(np.diff(p, axis=1), axis=0).sum()), 4)))
        print(f"     (A has let out {(hand['node'] - GRIP_NODE) * DL * 100:.1f} of "
              f"{N_RESERVE * DL * 100:.0f} cm of reserve)")
        if k in servo_targets:
            print(f"     (servo travel so far: A {servo['a']*100:+.1f} cm, B {servo['b']*100:+.1f} cm "
                  f"(+ = pulled out); strands A {a_tension:.1f} N, tail {tail_tension:.1f} N)")
        if args.stop_after is not None and k >= args.stop_after:
            print(f"     (stopping after stage {k}, as asked)")
            break

    print(f"\nclosest A ever came to B: {closest['run']*100:.1f} cm (limit {CLEARANCE*100:.0f} cm)")
    if unstable is None:
        a_tension, tail_tension = strand_tensions(rod.position_collection)
        print(f"final state: A's strand {a_tension:.1f} N, tail {tail_tension:.1f} N, buried stitch "
              f"{stitch_tension(rod.position_collection):.1f} N, both instruments holding")
    write_summary(args, unstable, closest, log_rows, stage_rows, rod, stitch, strand_tensions,
                  stitch_tension, winding, wall_start, cinch_tension)
    np.save("tension_knot_final.npy", snapshots[-1][0])
    np.save("tension_knot_stages.npy", np.array([s[0] for s in snapshots]))
    plot_stages(snapshots, stitch, wound)
    if unstable is not None:
        raise SystemExit("run stopped: thread went unstable (see report above)")
    return True


def write_summary(args, unstable, closest, log_rows, stage_rows, rod, stitch, strand_tensions,
                  stitch_tension, winding, wall_start, cinch_tension):
    with open("tension_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "stage", "a_tension", "tail_tension", "servo_a", "servo_b", "grip_force", "a_b_gap"])
        w.writerows(log_rows)
    p = rod.position_collection
    strands = {}
    for name, strand in (("tail", p[:, stitch[0]::-1]), ("long", p[:, stitch[-1]:])):
        det, dets, _ = knot_check.knot_determinant(knot_check.close_under(strand), closed=True)
        strands[name] = dict(det=int(det), agree=len(set(dets)) == 1)
    strain = np.linalg.norm(np.diff(p, axis=1), axis=0) / rod.rest_lengths - 1.0
    a_t, tail_t = strand_tensions(p)
    summary = dict(
        args={k: v for k, v in vars(args).items()}, stable=unstable is None,
        unstable_stage=None if unstable is None else int(unstable[0]),
        unstable_t=None if unstable is None else float(unstable[1]),
        last_stage=stage_rows[-1]["stage"] if stage_rows else None,
        final_a_tension=a_t, final_tail_tension=tail_t, final_stitch_tension=stitch_tension(p),
        min_clearance_m=float(closest["run"]), final_winding=winding(p), max_strain=float(strain.max()),
        strands=strands, target_tension=cinch_tension, stages=stage_rows,
        wall_s=round(time.time() - wall_start, 1))
    with open("run_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("wrote run_summary.json and tension_log.csv")


def plot_stages(snapshots, stitch, wound):
    tail = np.arange(0, stitch[0] + 1)
    for i, j, name, xlabel, ylabel, end_on in ((0, 1, "tension_side.png", "x (along B)", "y (up)", False),
                                               (1, 2, "tension_end.png", "y (up)", "z", True)):
        rows = int(np.ceil(len(snapshots) / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(21, 5 * rows), squeeze=False)
        for ax, (p, tip, label, a_pos, hand_node) in zip(axes.ravel(), snapshots):
            g = int(np.floor(hand_node))
            working, reserve = np.arange(stitch[-1], g + 1), np.arange(g, N_ELEMENTS + 1)
            if not end_on:
                draw_flaps(ax, p, wound)
            ax.plot(p[i, tail], p[j, tail], "-", color="darkorange", lw=3, label="tail")
            ax.plot(p[i, stitch], p[j, stitch], "-", color="black", lw=5, label="stitch (in tissue)")
            ax.plot(p[i, working], p[j, working], "-", color="tab:blue", lw=2, label="long strand")
            ax.plot(p[i, reserve], p[j, reserve], "--", color="tab:blue", lw=1, alpha=0.5,
                    label="reserve (beyond A's grip)")
            ax.plot(a_pos[i], a_pos[j], "o", color="tab:green", ms=9, label="A")
            if end_on:
                ax.add_patch(plt.Circle((tip[1], tip[2]), B_RADIUS, color="grey", alpha=0.6))
            else:
                ax.plot([tip[i] - B_LENGTH * B_AXIS[i], tip[i]], [tip[j] - B_LENGTH * B_AXIS[j], tip[j]],
                        "-", color="grey", lw=9, alpha=0.5, solid_capstyle="butt")
            ax.set_title(f"{label}\nwound {wound_gap(p, wound)*1000:+.1f} mm", fontsize=10)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        for ax in axes.ravel()[len(snapshots):]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(name, dpi=70)
        plt.close(fig)
    print("\nsaved tension_side.png and tension_end.png")


if __name__ == "__main__":
    run()
