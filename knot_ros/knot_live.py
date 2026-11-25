"""Ties the knot live: PyElastica steps the thread and both arms follow by IK."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float64, String  # noqa: E402
from visualization_msgs.msg import MarkerArray  # noqa: E402

import scene  # noqa: E402
from kin import Chain  # noqa: E402
from plan_motion import A_AXIS_SIM, tool_pose, solve_from_scratch, capsules, segment_distance, pedestal_box  # noqa: E402
from knot_replay import scene_markers  # noqa: E402


class KnotLive(Node):
    def __init__(self, urdf_path, stop_on_clash):
        super().__init__("knot_live")
        urdf = open(urdf_path).read()
        self.chains = {arm: Chain(urdf, "world", f"{arm}tool0") for arm in ("a_", "b_")}
        self.q = {arm: None for arm in self.chains}
        self.joint_names = []
        for arm, chain in self.chains.items():
            self.joint_names += chain.names + [f"{arm}jaw_up_joint", f"{arm}jaw_down_joint"]
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/knot_markers", 10)
        self.value_pubs = {k: self.create_publisher(Float64, f"/knot/{k}", 10)
                           for k in ("tension_a", "tension_tail", "arm_clearance")}
        self.stage_pub = self.create_publisher(String, "/knot/stage", 10)
        self.stop_on_clash = stop_on_clash
        self.closest = (np.inf, 0.0, "")
        self.clashes = 0
        self.a_hinge = None
        self.started = None
        self.jaws = {"a_": scene.JAW_OPEN, "b_": scene.JAW_OPEN}
        self.last_t = None
        self.pedestal = pedestal_box()

    def solve(self, arm, hinge, axis, where=""):
        """Warm-started IK; from scratch only if that fails."""
        chain = self.chains[arm]
        target = tool_pose(np.asarray(hinge, dtype=float), np.asarray(axis, dtype=float))
        if self.q[arm] is not None:
            q, pos_err, rot_err = chain.ik(target, self.q[arm])
            if pos_err <= scene.IK_POS_TOL and rot_err <= scene.IK_ROT_TOL:
                self.q[arm] = q
                return q
        q = solve_from_scratch(chain, target)
        if q is None:
            raise SystemExit(f"arm {arm[0].upper()} cannot reach its instrument {where} - "
                             f"move BASE_{arm[0].upper()} in scene.py")
        self.q[arm] = q
        return q

    def arm_clearance(self, b_radius):
        caps_a = capsules(self.chains["a_"], self.q["a_"], scene.SHAFT_RADIUS)
        caps_b = capsules(self.chains["b_"], self.q["b_"], b_radius)
        best = (np.inf, "", "")
        for c1 in caps_a:
            for c2 in caps_b:
                d = segment_distance(c1[0], c1[1], c2[0], c2[1]) - c1[2] - c2[2]
                if d < best[0]:
                    best = (d, c1[3], c2[3])
        return best

    def precheck(self, sim, a_pull=0.20, b_pull=0.10):
        """Check both arms can reach the whole planned path before running the physics."""
        pos, _, _ = sim.build_thread()
        a_start = pos[:, sim.GRIP_NODE].copy()
        plan = sim.Plan(sim.stages(a_start), a_start)
        worst = np.inf
        for a_extra, b_extra in ((0.0, 0.0), (a_pull, b_pull)):
            plan.a_offset = a_extra * np.array([1.0, 0.0, 0.0])
            plan.b_offset = b_extra * np.array([-1.0, 0.0, 0.0])
            self.q = {arm: None for arm in self.chains}
            for t in np.arange(0.0, plan.ends[-1], 0.1):
                where = f"at t = {t:.1f} s ({plan.stages[plan.stage_at(t)[0]][4]})"
                self.solve("a_", scene.to_world(plan.a_pos(t)), scene.direction_to_world(A_AXIS_SIM), where)
                self.solve("b_", scene.to_world(plan.b_tip(t)), scene.direction_to_world(sim.B_AXIS), where)
                worst = min(worst, self.arm_clearance(sim.B_RADIUS)[0])
        self.q = {arm: None for arm in self.chains}
        self.get_logger().info(f"precheck: both arms reach the whole planned path (and the servo pulls); "
                               f"arms never closer than {worst*100:.1f} cm")
        if worst < scene.CLEARANCE:
            raise SystemExit(f"planned path brings the arms within {worst*100:.1f} cm - "
                             f"limit {scene.CLEARANCE*100:.0f} cm; adjust scene.py")

    def on_state(self, s):
        """Called by tension_knot.run() as the simulation advances."""
        if self.started is None:
            self.started = (s["t"], s["wall"])
        b_hinge = scene.to_world(s["b_tip"])
        b_axis = scene.direction_to_world(s["b_axis"])
        if s["a_pos"] is not None:
            self.a_hinge = scene.to_world(s["a_pos"])
        where = f"at t = {s['t']:.2f} s ({s['label']})"
        q_a = self.solve("a_", self.a_hinge, scene.direction_to_world(A_AXIS_SIM), where)
        q_b = self.solve("b_", b_hinge, b_axis, where)

        gap = self.arm_clearance(s["b_radius"])
        if gap[0] < self.closest[0]:
            self.closest = (gap[0], s["t"], f"A:{gap[1]} - B:{gap[2]}")
        if gap[0] < scene.CLEARANCE:
            self.clashes += 1
            if self.clashes == 1 or self.clashes % 50 == 0:
                self.get_logger().warn(f"arms {gap[0]*100:.1f} cm apart (A:{gap[1]} - B:{gap[2]}) {where}")
            if self.stop_on_clash:
                raise SystemExit(f"stopped: arms within {scene.CLEARANCE*100:.0f} cm {where}")

        step = scene.JAW_SPEED * (s["t"] - self.last_t if self.last_t is not None else 1.0)
        self.last_t = s["t"]
        for arm, wanted in (("a_", 0.0 if s["a_pos"] is not None else scene.JAW_OPEN),
                            ("b_", 0.0 if s["gripping"] else scene.JAW_OPEN)):
            self.jaws[arm] += float(np.clip(wanted - self.jaws[arm], -step, step))
        stamp = self.get_clock().now().to_msg()
        state = JointState()
        state.header.stamp = stamp
        state.name = self.joint_names
        state.position = ([float(v) for v in q_a] + [self.jaws["a_"]] * 2
                          + [float(v) for v in q_b] + [self.jaws["b_"]] * 2)
        self.joint_pub.publish(state)

        sim_time, wall_time = s["t"] - self.started[0], s["wall"] - self.started[1]
        realtime = sim_time / wall_time if wall_time > 1e-6 else 0.0
        moment = dict(thread=scene.to_world(s["thread"]), hand_node=s["hand_node"], stitch=s["stitch"],
                      n_elements=s["n_elements"], thread_radius=s["thread_radius"], b_tip=b_hinge, b_axis=b_axis,
                      b_length=s["b_length"], b_radius=s["b_radius"], pedestal_lo=self.pedestal[0],
                      pedestal_hi=self.pedestal[1], left_exit=s["left_exit"], right_exit=s["right_exit"],
                      edge_bite=s["edge_bite"],
                      caption=(f"LIVE  stage {s['stage']}: {s['label']}   t = {s['t']:.2f} s   "
                               f"A {s['a_tension']:.1f} N  tail {s['tail_tension']:.1f} N   "
                               f"arms {gap[0]*100:.1f} cm apart   x{realtime:.3f} real time"))
        self.marker_pub.publish(scene_markers(stamp, moment))
        for topic, value in (("tension_a", s["a_tension"]), ("tension_tail", s["tail_tension"]),
                             ("arm_clearance", gap[0])):
            msg = Float64()
            msg.data = float(value)
            self.value_pubs[topic].publish(msg)
        msg = String()
        msg.data = s["label"]
        self.stage_pub.publish(msg)


def main():
    try:
        from rclpy.utilities import remove_ros_args
        argv = remove_ros_args(sys.argv)[1:]
    except Exception:
        argv = sys.argv[1:]
    sim_args = []
    if "--" in argv:
        split = argv.index("--")
        argv, sim_args = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=os.path.join(HERE, "dual_ur7e_grasper.urdf"))
    parser.add_argument("--sim-dir", default=os.path.dirname(HERE),
                        help="folder with tension_knot.py and its helpers; outputs go there too")
    parser.add_argument("--live-dt", type=float, default=0.005, help="simulated seconds between updates")
    parser.add_argument("--stop-on-clash", action="store_true",
                        help=f"stop if the arms come within {scene.CLEARANCE*100:.0f} cm")
    args = parser.parse_args(argv)
    if not os.path.exists(args.urdf):
        raise SystemExit(f"{args.urdf} not found - run dual_arm.py first")
    sim_dir = os.path.abspath(args.sim_dir)
    sys.path.insert(0, sim_dir)
    os.chdir(sim_dir)
    import tension_knot

    rclpy.init()
    node = KnotLive(args.urdf, args.stop_on_clash)
    finished = False
    try:
        node.precheck(tension_knot)
        finished = tension_knot.run(sim_args, live_hook=node.on_state, live_dt=args.live_dt)
    except SystemExit as e:
        node.get_logger().error(str(e))
    finally:
        if np.isfinite(node.closest[0]):
            node.get_logger().info(
                f"arms never closer than {node.closest[0]*100:.1f} cm ({node.closest[2]} at "
                f"t = {node.closest[1]:.2f} s); {node.clashes} updates under "
                f"{scene.CLEARANCE*100:.0f} cm; run {'finished' if finished else 'did not finish'}")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
