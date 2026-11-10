"""Scripted pick-and-place expert.

Key design choice: the expert has NO memory. expert(obs) is a pure function of the
current observation. There is no phase counter or timer. That matters for week 6:
  * a readout imitating it is not being asked to recover hidden state, and
  * it can be queried at any state the imitator wanders into, which is what
    DAgger needs. It also re-grasps by itself if the cube is dropped.
The phase name is returned only as a label for logging and plots.
"""
import numpy as np
from panda_env import MAX_DELTA, CUBE_HALF

HOVER_Z = 0.13        # m, gripper height while travelling
GRASP_Z = 0.03        # m, gripper height for grasping
ALIGN_XY = 0.012      # m, xy alignment needed to start descending
STRADDLE_XY = 0.02    # m, xy offset at which the cube is still safely between the fingers
CLOSE_XY = 0.015      # m, xy offset allowed when closing
KP = 0.6              # fraction of the remaining error covered per step
HELD_W = CUBE_HALF * 2 + 0.006   # gripper narrower than this means it is on the cube
RELEASE_CUBE_Z = CUBE_HALF + 0.006   # open once the cube is this low over the zone
OPEN, CLOSE = 1.0, -1.0

PHASES = ["reach", "descend", "close", "lift", "carry", "lower", "release", "retreat"]


def move(tcp, goal):
    return np.clip(KP * (goal - tcp), -MAX_DELTA, MAX_DELTA)


def expert(o):
    """o: observation vector from PandaPickPlace.obs(). Returns (action, phase name)."""
    tcp, width, wvel = o[:3], o[6], o[7]
    cube, goal = o[8:11], o[14:16]
    rel = cube - tcp
    rel_xy = np.linalg.norm(rel[:2])
    d_goal = np.linalg.norm(cube[:2] - goal)
    straddled = rel_xy < STRADDLE_XY and abs(rel[2]) < 0.025     # cube is between the fingers
    # A lifted cube counts as held even if the fingers jiggle under acceleration.
    gripping = straddled and width < HELD_W and (abs(wvel) < 0.05 or cube[2] > CUBE_HALF + 0.01)

    # Cube is low and at the goal: let go and leave.
    if d_goal < 0.03 and cube[2] < RELEASE_CUBE_Z:
        if width < CUBE_HALF * 2 + 0.02 and straddled:
            return np.r_[0, 0, 0, OPEN], "release"
        return np.r_[move(tcp, np.r_[tcp[:2], HOVER_Z]), OPEN], "retreat"

    # Carrying. All goals are written for the cube, using the measured cube-in-gripper
    # offset (rel), so a cube held slightly off-centre is still placed on the goal.
    if gripping:
        if d_goal < 0.008 or (d_goal < 0.02 and tcp[2] < HOVER_Z - 0.01):
            return np.r_[move(tcp, np.r_[goal - rel[:2], CUBE_HALF + 0.003 - rel[2]]), CLOSE], "lower"
        if tcp[2] < HOVER_Z - 0.03:
            return np.r_[move(tcp, np.r_[tcp[:2], HOVER_Z]), CLOSE], "lift"
        return np.r_[move(tcp, np.r_[goal - rel[:2], HOVER_Z]), CLOSE], "carry"

    # Fingers around the cube at grasp height, or already closing on it.
    if rel_xy < CLOSE_XY and tcp[2] < GRASP_Z + 0.012:
        return np.r_[0, 0, 0, CLOSE], "close"
    if straddled and width < HELD_W + 0.01:
        return np.r_[0, 0, 0, CLOSE], "close"

    # Not holding it: open, then funnel down onto the cube. Tolerances widen once the
    # fingers are around the cube, so small tracking errors do not abort a descent.
    low = tcp[2] < 0.09
    if rel_xy < (STRADDLE_XY if low else ALIGN_XY):
        return np.r_[move(tcp, np.r_[cube[:2], GRASP_Z]), OPEN], "descend"
    if tcp[2] < HOVER_Z - 0.03:
        return np.r_[move(tcp, np.r_[tcp[:2], HOVER_Z]), OPEN], "reach"
    return np.r_[move(tcp, np.r_[cube[:2], HOVER_Z]), OPEN], "reach"
