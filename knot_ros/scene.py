"""Shared geometry for the two-arm scene (sim has y up, ROS world z up)."""
import numpy as np

PEDESTAL_HEIGHT = 0.30
SIM_TO_WORLD = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])
WORLD_OFFSET = np.array([0.0, 0.0, PEDESTAL_HEIGHT])


def to_world(p):
    """Point (3,) or points (3, n) from the simulation into ROS world."""
    p = np.asarray(p, dtype=float)
    return SIM_TO_WORLD @ p + (WORLD_OFFSET if p.ndim == 1 else WORLD_OFFSET[:, None])


def direction_to_world(v):
    return SIM_TO_WORLD @ np.asarray(v, dtype=float)


HINGE_OFFSET = 0.33
JAW_LENGTH = 0.020
GRASPER_MOUNT_XYZ = (0.0, 0.0, 0.33)
GRASPER_MOUNT_RPY = (-1.5708, 0.0, 0.0)
SHAFT_RADIUS = 0.005
JAW_OPEN = 0.5                # rad per jaw
JAW_SPEED = 3.0               # rad/s

BASE_B = np.array([-0.85, 0.0, 0.0])
YAW_B = np.pi
BASE_A = np.array([0.95, 0.0, 0.0])
YAW_A = 0.0

CLEARANCE = 0.03
LINK_RADIUS = 0.055
PAD_X = (-0.115, 0.115)
PAD_Y = (-0.105, 0.045)          # world y: along the wound
PAD_THICKNESS = 0.015
PAD_SKIN, PAD_FAT = 0.002, 0.006  # muscle = the rest
PEDESTAL_X, PEDESTAL_Y = PAD_X, PAD_Y
IK_POS_TOL = 1e-3
IK_ROT_TOL = np.radians(1.0)
