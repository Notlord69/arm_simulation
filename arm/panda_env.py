"""Week 4: Panda pick-and-place environment (no brain involved).

Scene:      Panda bolted to a table top at z = 0, one free cube, one flat target zone.
Action:     [dx, dy, dz, grip], one per policy step (25 Hz, 40 ms).
            dx, dy, dz: metres the gripper target moves this step, clipped to +-MAX_DELTA.
            grip: > 0 open, <= 0 close.
Controller: damped least-squares IK with a null-space posture term, run at 500 Hz,
            driving the model's joint position servos. Gripper orientation is held
            pointing straight down, because the action has no rotation terms.
Success:    measured from physics only. It knows nothing about the expert.
"""
from pathlib import Path
import numpy as np
import mujoco

MENAGERIE = Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "franka_emika_panda"

POLICY_DT = 0.04          # s, one action every 40 ms (matches the week 5 decoder window)
MAX_DELTA = 0.02          # m per policy step, so 0.5 m/s peak
HOME_Q = np.array([0, -0.2, 0, -2.3, 0, 2.1, np.pi / 4])

CUBE_HALF = 0.025         # m
CUBE_MASS = 0.08          # kg
SPAWN_X = (0.35, 0.60)    # m, cube spawn rectangle
SPAWN_Y = (-0.25, 0.15)
TARGET_XY = np.array([0.45, 0.28])
ZONE_RADIUS = 0.05        # m, drawn on the table
WORKSPACE_LO = np.array([0.15, -0.45, 0.012])   # the gripper target is clamped to this box
WORKSPACE_HI = np.array([0.75, 0.45, 0.50])
GRIP_SCALE = 10.0
MAX_STEPS = 250           # 10 s of brain time per episode

OBS_NAMES = (["tcp_x", "tcp_y", "tcp_z", "tcp_vx", "tcp_vy", "tcp_vz", "grip_width", "grip_vel",
              "cube_x", "cube_y", "cube_z", "cube_vx", "cube_vy", "cube_vz",
              "goal_x", "goal_y",
              "cube_rel_x", "cube_rel_y", "cube_rel_z",        # cube minus gripper
              "goal_rel_x", "goal_rel_y",                      # goal minus cube
              "lag_x", "lag_y", "lag_z"])                      # controller target minus gripper
ACT_NAMES = ["dx", "dy", "dz", "grip"]


def build_model():
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / "scene.xml"))
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC   # better for friction grasps
    spec.option.impratio = 10
    for key in list(spec.keys):                         # old keyframe has no cube in it
        spec.delete(key)

    floor = spec.geom("floor")
    floor.pos = [0, 0, -0.75]
    w = spec.worldbody
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.75, 0.75, 0.02],
               pos=[0.3, 0, -0.02], rgba=[0.55, 0.42, 0.3, 1], friction=[0.8, 0.005, 0.0001])
    w.add_geom(name="zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[ZONE_RADIUS, 0.0005, 0],
               pos=[TARGET_XY[0], TARGET_XY[1], 0.0005], rgba=[0.1, 0.8, 0.2, 0.6],
               contype=0, conaffinity=0)                # painted on: it has no collision

    cube = w.add_body(name="cube", pos=[0.5, 0, CUBE_HALF])
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                  mass=CUBE_MASS, rgba=[0.85, 0.15, 0.15, 1], condim=4,
                  friction=[1.0, 0.02, 0.001])

    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005] * 3,
                               rgba=[1, 1, 0, 0.5], group=4)
    m = spec.compile()
    # Grip strength. The Menagerie gripper servo only squeezes with 100 N/m * finger opening,
    # about 2.5 N of tendon force on a 5 cm cube, and the cube slips and swings in the fingers.
    # The real Panda grips with 30-70 N. Scaling the servo 10x gives about 25 N.
    m.actuator_gainprm[7, 0] *= GRIP_SCALE
    m.actuator_biasprm[7, 1:3] *= GRIP_SCALE
    return m


class PandaPickPlace:
    def __init__(self, model=None):
        self.m = model if model is not None else build_model()
        self.d = mujoco.MjData(self.m)
        m = self.m
        ids = lambda t, n: mujoco.mj_name2id(m, t, n)
        self.tcp = ids(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.cube_body = ids(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = ids(mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self.q_adr = np.array([m.jnt_qposadr[ids(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")] for i in range(1, 8)])
        self.v_adr = np.array([m.jnt_dofadr[ids(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")] for i in range(1, 8)])
        self.f_adr = np.array([m.jnt_qposadr[ids(mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{i}")] for i in (1, 2)])
        self.fv_adr = np.array([m.jnt_dofadr[ids(mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{i}")] for i in (1, 2)])
        self.c_adr = m.jnt_qposadr[ids(mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")]
        self.cv_adr = m.jnt_dofadr[ids(mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")]
        self.kp = m.actuator_gainprm[:7, 0].copy()
        self.kd = -m.actuator_biasprm[:7, 2].copy()
        self.ik = mujoco.MjData(m)      # kinematics-only copy used by the IK
        self.q_lo, self.q_hi = m.jnt_range[[i for i in range(m.njnt) if m.jnt_qposadr[i] in self.q_adr]].T
        self.n_sub = int(round(POLICY_DT / m.opt.timestep))
        self.finger_geoms = {i for i in range(m.ngeom)
                             if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i])
                             in ("left_finger", "right_finger", "hand")}
        self._jacp, self._jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    # ---------- episode ----------
    def reset(self, seed=None, cube_xy=None, cube_yaw=0.0):
        """Seeded reset. Same seed gives the same cube position and start pose."""
        m, d = self.m, self.d
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(m, d)
        if cube_xy is None:
            cube_xy = (rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y))
        d.qpos[self.q_adr] = HOME_Q + rng.uniform(-0.03, 0.03, 7)
        d.qpos[self.f_adr] = 0.04
        d.qpos[self.c_adr:self.c_adr + 3] = [cube_xy[0], cube_xy[1], CUBE_HALF]
        d.qpos[self.c_adr + 3:self.c_adr + 7] = [np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)]
        mujoco.mj_forward(m, d)
        self.ik.qpos[:] = d.qpos
        self.q_cmd = d.qpos[self.q_adr].copy()
        self.q_home = self.q_cmd.copy()
        self.x_cmd = d.site_xpos[self.tcp].copy()
        self.grip_cmd = 255.0
        self.down = np.array([0.0, 1.0, 0.0, 0.0])      # (w,x,y,z): gripper z axis points at the table
        d.ctrl[:7] = self.q_cmd + d.qfrc_bias[self.v_adr] / self.kp
        d.ctrl[7] = self.grip_cmd
        self.t = 0
        self.cube_xy0 = np.array(cube_xy, float)
        return self.obs()

    def step(self, action):
        """Apply one action for 40 ms of physics."""
        a = np.asarray(action, float)
        delta = np.clip(a[:3], -MAX_DELTA, MAX_DELTA)
        self.grip_cmd = 255.0 if a[3] > 0 else 0.0
        x0 = self.x_cmd.copy()
        x1 = np.clip(x0 + delta, WORKSPACE_LO, WORKSPACE_HI)
        for i in range(self.n_sub):
            self.x_cmd = x0 + (x1 - x0) * (i + 1) / self.n_sub   # ramp, so no step in the target
            self._ik_step((x1 - x0) / POLICY_DT)
            self.d.ctrl[7] = self.grip_cmd
            mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self.t += 1
        return self.obs()

    def _ik_step(self, v_ff):
        """One 500 Hz step: task-space error to joint velocity to joint position command.

        IK runs on the COMMANDED joint angles (self.ik, a kinematics-only copy), not the
        measured ones. The servos have a 0.1 s lag (kd/kp = 450/4500), so closing the IK
        loop through the measured angles at this gain makes the arm oscillate.
        Two feed-forward terms then let the servo follow the command closely:
        gravity (qfrc_bias / kp) and velocity (dq * kd / kp).
        """
        m, d, k = self.m, self.d, self.ik
        dt = m.opt.timestep
        k.qpos[self.q_adr] = self.q_cmd
        mujoco.mj_kinematics(m, k); mujoco.mj_comPos(m, k)
        mujoco.mj_jacSite(m, k, self._jacp, self._jacr, self.tcp)
        J = np.vstack([self._jacp[:, self.v_adr], self._jacr[:, self.v_adr]])
        ep = self.x_cmd - k.site_xpos[self.tcp]
        qcur = np.empty(4); mujoco.mju_mat2Quat(qcur, k.site_xmat[self.tcp])
        qinv = np.empty(4); mujoco.mju_negQuat(qinv, qcur)
        qerr = np.empty(4); mujoco.mju_mulQuat(qerr, self.down, qinv)
        er = np.empty(3); mujoco.mju_quat2Vel(er, qerr, 1.0)
        twist = np.r_[v_ff + 30.0 * ep, 30.0 * er]
        JJt = J @ J.T + 1e-3 * np.eye(6)                   # damping keeps it safe near singularities
        Jpinv = J.T @ np.linalg.solve(JJt, np.eye(6))
        dq = Jpinv @ twist + (np.eye(7) - Jpinv @ J) @ (2.0 * (self.q_home - self.q_cmd))
        dq = np.clip(dq, -2.0, 2.0)
        self.q_cmd = np.clip(self.q_cmd + dq * dt, self.q_lo, self.q_hi)
        d.ctrl[:7] = self.q_cmd + (self.kd * dq + d.qfrc_bias[self.v_adr]) / self.kp

    # ---------- reading state ----------
    def tcp_pos(self): return self.d.site_xpos[self.tcp].copy()
    def cube_pos(self): return self.d.xpos[self.cube_body].copy()
    def grip_width(self): return float(self.d.qpos[self.f_adr].sum())

    def tcp_vel(self):
        v = np.empty(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, self.tcp, v, 0)
        return v[3:]

    def cube_vel(self): return self.d.qvel[self.cv_adr:self.cv_adr + 3].copy()

    def obs(self):
        x, c = self.tcp_pos(), self.cube_pos()
        return np.r_[x, self.tcp_vel(), self.grip_width(), self.d.qvel[self.fv_adr].sum(),
                     c, self.cube_vel(), TARGET_XY, c - x, TARGET_XY - c[:2], self.x_cmd - x]

    def cube_touching_gripper(self):
        d = self.d
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if (g1 == self.cube_geom and g2 in self.finger_geoms) or \
               (g2 == self.cube_geom and g1 in self.finger_geoms):
                return True
        return False

    def info(self):
        """Ground-truth outcome. This is what the success rate is computed from."""
        c = self.cube_pos()
        err = float(np.linalg.norm(c[:2] - TARGET_XY))
        in_zone = err < ZONE_RADIUS
        at_rest = abs(c[2] - CUBE_HALF) < 0.003 and np.linalg.norm(self.cube_vel()) < 0.01
        released = not self.cube_touching_gripper()
        clear = self.tcp_pos()[2] - c[2] > 0.06
        return dict(success=bool(in_zone and at_rest and released and clear),
                    place_error=err, in_zone=bool(in_zone), at_rest=bool(at_rest),
                    released=bool(released), clear=bool(clear))
