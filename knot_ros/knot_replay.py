"""Replays knot_motion.npz in RViz: joint states plus thread, instrument and tissue markers."""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import scene


def quaternion_z_to(v):
    """(x, y, z, w) turning +z onto direction v."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ v)
    if c < -0.999999:
        return 1.0, 0.0, 0.0, 0.0
    axis = np.cross(z, v)
    s = np.sqrt(2.0 * (1.0 + c))
    return axis[0] / s, axis[1] / s, axis[2] / s, s / 2.0


def new_marker(stamp, marker_id, kind, ns, rgba, scale):
    m = Marker()
    m.header.frame_id = "world"
    m.header.stamp = stamp
    m.ns, m.id, m.type, m.action = ns, marker_id, kind, Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = scale
    m.color.r, m.color.g, m.color.b, m.color.a = rgba
    return m


def line(stamp, marker_id, points, rgba, width):
    m = new_marker(stamp, marker_id, Marker.LINE_STRIP, "thread", rgba, (width, 0.0, 0.0))
    m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points.T]
    return m


def scene_markers(stamp, moment, show_reserve=False, min_width=0.0015):
    """Markers for one moment (used by the replay and the live node)."""
    thread = np.asarray(moment["thread"], dtype=float)
    stitch = moment["stitch"]
    width = max(2.0 * float(moment["thread_radius"]), float(min_width))
    hand = int(np.floor(moment["hand_node"]))
    out = MarkerArray()
    out.markers.append(line(stamp, 0, thread[:, :stitch[0] + 1], (1.0, 0.55, 0.0, 1.0), width))
    out.markers.append(line(stamp, 1, thread[:, stitch], (0.05, 0.05, 0.05, 1.0), width))
    out.markers.append(line(stamp, 2, thread[:, stitch[-1]:hand + 1], (0.12, 0.47, 0.71, 1.0), width))
    if show_reserve and hand < int(moment["n_elements"]):
        out.markers.append(line(stamp, 3, thread[:, hand:], (0.12, 0.47, 0.71, 0.3), width * 0.6))

    tip, axis = np.asarray(moment["b_tip"], dtype=float), np.asarray(moment["b_axis"], dtype=float)
    length, radius = float(moment["b_length"]), float(moment["b_radius"])
    sleeve = new_marker(stamp, 10, Marker.CYLINDER, "instrument_B_sleeve", (0.6, 0.6, 0.6, 0.25),
                        (2 * radius, 2 * radius, length))
    centre = tip - 0.5 * length * axis / np.linalg.norm(axis)
    sleeve.pose.position.x, sleeve.pose.position.y, sleeve.pose.position.z = map(float, centre)
    q = quaternion_z_to(axis)
    sleeve.pose.orientation.x, sleeve.pose.orientation.y, sleeve.pose.orientation.z, sleeve.pose.orientation.w = q
    out.markers.append(sleeve)

    top = scene.PEDESTAL_HEIGHT - float(moment["thread_radius"])
    th = float(moment.get("tissue_thickness", scene.PAD_THICKNESS))
    skin = min(scene.PAD_SKIN, th)
    fat = (th - skin) * scene.PAD_FAT / max(scene.PAD_THICKNESS - scene.PAD_SKIN, 1e-9)
    (x_lo, x_hi), (y_lo, y_hi) = scene.PAD_X, scene.PAD_Y
    stand = new_marker(stamp, 20, Marker.CUBE, "tissue", (0.55, 0.55, 0.58, 1.0),
                       (x_hi - x_lo, y_hi - y_lo, top - th))
    stand.pose.position.x, stand.pose.position.y, stand.pose.position.z = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2, (top - th) / 2
    out.markers.append(stand)
    left = float(thread[0, int(moment["left_exit"])])
    right = float(thread[0, int(moment["right_exit"])])
    bite = float(moment["edge_bite"])
    layers = ((top - skin, skin, (0.93, 0.74, 0.64, 1.0)),            # skin
              (top - skin - fat, fat, (0.98, 0.88, 0.56, 1.0)),       # fat
              (top - th, th - skin - fat, (0.72, 0.28, 0.26, 1.0)))   # muscle
    marker_id = 21
    for z0, height, rgba in layers:
        if height <= 0:
            continue
        for x0, x1 in ((x_lo, left + bite), (right - bite, x_hi)):
            box = new_marker(stamp, marker_id, Marker.CUBE, "tissue", rgba, (max(x1 - x0, 1e-3), y_hi - y_lo, height))
            box.pose.position.x, box.pose.position.y, box.pose.position.z = (x0 + x1) / 2, (y_lo + y_hi) / 2, z0 + height / 2
            out.markers.append(box)
            marker_id += 1
    hi = np.array([x_hi, y_hi, scene.PEDESTAL_HEIGHT])

    caption = new_marker(stamp, 30, Marker.TEXT_VIEW_FACING, "caption", (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.035))
    caption.pose.position.z = float(hi[2]) + 0.35
    caption.text = moment["caption"]
    out.markers.append(caption)
    return out


class KnotReplay(Node):
    def __init__(self):
        super().__init__("knot_replay")
        self.declare_parameter("motion", "knot_motion.npz")
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("show_reserve", False)
        self.declare_parameter("hold_end", 2.0)
        self.declare_parameter("thread_width", 0.0015)
        self.declare_parameter("tissue_thickness", scene.PAD_THICKNESS)  # drawn pad thickness (m)
        data = np.load(self.get_parameter("motion").value)
        self.motion = {k: data[k] for k in data.files}
        self.joint_names = [str(n) for n in self.motion["joint_names"]]
        self.labels = [str(s) for s in self.motion["labels"]]
        self.n_frames = len(self.motion["t"])
        self.frame = 0
        self.held = 0
        speed = float(self.get_parameter("speed").value)
        period = float(self.motion["frame_dt"]) / max(speed, 1e-3)
        self.hold_frames = int(float(self.get_parameter("hold_end").value) / period)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/knot_markers", 10)
        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(
            f"replaying {self.n_frames} frames ({self.motion['t'][-1] - self.motion['t'][0]:.1f} s simulated) "
            f"at {speed}x; arms never closer than {float(self.motion['min_clearance'])*100:.1f} cm; thread "
            f"{2000*float(self.motion['r_rod']):.2f} mm across, drawn "
            f"{1000*max(2*float(self.motion['r_rod']), float(self.get_parameter('thread_width').value)):.2f} mm")

    def markers(self, i):
        m = self.motion
        k = int(m["stage"][i])
        moment = dict(thread=m["thread"][i], hand_node=m["hand_g"][i], stitch=m["bite"], n_elements=m["n"],
                      thread_radius=m["r_rod"], b_tip=m["b_tip"][i], b_axis=m["b_u"][i], b_length=m["b_len"],
                      b_radius=m["b_r"], pedestal_lo=m["pedestal_lo"], pedestal_hi=m["pedestal_hi"],
                      left_exit=m["wound_iL"], right_exit=m["wound_iR"], edge_bite=m["edge_bite"],
                      tissue_thickness=float(self.get_parameter("tissue_thickness").value),
                      caption=(f"stage {k}: {self.labels[k]}   t = {m['t'][i]:.2f} s   "
                               f"wound {float(m['wound'][i])*1000:+.1f} mm"))
        return scene_markers(self.get_clock().now().to_msg(), moment,
                             bool(self.get_parameter("show_reserve").value),
                             float(self.get_parameter("thread_width").value))

    def tick(self):
        state = JointState()
        state.header.stamp = self.get_clock().now().to_msg()
        state.name = self.joint_names
        state.position = [float(v) for v in self.motion["q"][self.frame]]
        self.joint_pub.publish(state)
        self.marker_pub.publish(self.markers(self.frame))
        if self.frame < self.n_frames - 1:
            self.frame += 1
        elif self.get_parameter("loop").value:
            self.held += 1
            if self.held >= self.hold_frames:
                self.frame, self.held = 0, 0


def main():
    rclpy.init()
    node = KnotReplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
