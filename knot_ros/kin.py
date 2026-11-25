"""Forward and inverse kinematics read straight from a URDF (no MoveIt)."""
import xml.etree.ElementTree as ET
import numpy as np


def rpy_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rotation_about(axis, angle):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def numbers(text, default=0.0):
    return [float(v) for v in text.split()] if text else [default] * 3


class Chain:
    """The joints between two links of a URDF, root first."""

    def __init__(self, urdf_xml, root, tip):
        robot = ET.fromstring(urdf_xml)
        joint_into = {}
        for j in robot.findall("joint"):
            origin, axis, limit = j.find("origin"), j.find("axis"), j.find("limit")
            joint_into[j.find("child").get("link")] = dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                xyz=np.array(numbers(origin.get("xyz") if origin is not None else None)),
                R=rpy_matrix(*numbers(origin.get("rpy") if origin is not None else None)),
                axis=np.array(numbers(axis.get("xyz"))) if axis is not None else np.array([1.0, 0.0, 0.0]),
                lower=float(limit.get("lower", -2 * np.pi)) if limit is not None else -2 * np.pi,
                upper=float(limit.get("upper", 2 * np.pi)) if limit is not None else 2 * np.pi)
        joints, link = [], tip
        while link != root:
            if link not in joint_into:
                raise ValueError(f"no joint chain from {root} to {tip} (stuck at {link})")
            j = joint_into[link]
            joints.append(dict(j, child=link))
            link = j["parent"]
        self.joints = joints[::-1]
        moving = [j for j in self.joints if j["type"] in ("revolute", "continuous")]
        self.names = [j["name"] for j in moving]
        self.lower = np.array([j["lower"] if j["type"] == "revolute" else -4 * np.pi for j in moving])
        self.upper = np.array([j["upper"] if j["type"] == "revolute" else 4 * np.pi for j in moving])
        self.n = len(moving)

    def frames(self, q):
        """World pose of every link, plus each moving joint's world axis and origin."""
        T = np.eye(4)
        poses, axes, origins = [], [], []
        angles = iter(q)
        for j in self.joints:
            step = np.eye(4)
            step[:3, :3], step[:3, 3] = j["R"], j["xyz"]
            T = T @ step
            if j["type"] in ("revolute", "continuous"):
                axes.append(T[:3, :3] @ (j["axis"] / np.linalg.norm(j["axis"])))
                origins.append(T[:3, 3].copy())
                turn = np.eye(4)
                turn[:3, :3] = rotation_about(j["axis"], next(angles))
                T = T @ turn
            poses.append(T.copy())
        return poses, np.array(axes), np.array(origins)

    def fk(self, q):
        return self.frames(q)[0][-1]

    def jacobian(self, q):
        poses, axes, origins = self.frames(q)
        tip = poses[-1][:3, 3]
        J = np.zeros((6, self.n))
        for i in range(self.n):
            J[:3, i] = np.cross(axes[i], tip - origins[i])
            J[3:, i] = axes[i]
        return J, poses[-1]

    @staticmethod
    def pose_error(T, target):
        position = target[:3, 3] - T[:3, 3]
        rotation = 0.5 * sum(np.cross(T[:3, k], target[:3, k]) for k in range(3))
        return np.concatenate([position, rotation])

    def ik(self, target, start, iters=200, damping=0.02):
        """Damped least squares from `start`; returns (q, position error, rotation error)."""
        q = np.array(start, dtype=float)
        for _ in range(iters):
            J, T = self.jacobian(q)
            e = self.pose_error(T, target)
            if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-5:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            size = np.linalg.norm(dq)
            if size > 0.3:
                dq *= 0.3 / size
            q = np.clip(q + dq, self.lower, self.upper)
        e = self.pose_error(self.fk(q), target)
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
