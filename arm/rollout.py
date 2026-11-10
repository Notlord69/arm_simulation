"""Run one episode with any policy. Shared by evaluation, data collection and imitation tests."""
import numpy as np
from panda_env import MAX_STEPS, MAX_DELTA
from expert import expert, PHASES

HOLD_STEPS = 10   # success must hold for this many steps in a row (0.4 s), so a bouncing cube fails


def rollout(env, policy=None, seed=None, log=False, action_noise=0.0, obs_noise=0.0,
            noise_seed=None, max_steps=MAX_STEPS, **reset_kw):
    """policy(obs) returns an action. Default: the expert.
    action_noise: std of Gaussian noise (m) added to the EXECUTED dx, dy, dz. The logged
    action stays the clean expert label, so the data shows how to recover from errors.
    obs_noise: std of Gaussian noise (m) added to the cube position the policy sees.
    """
    rng = np.random.default_rng(noise_seed if noise_seed is not None else seed)
    o = env.reset(seed, **reset_kw)
    held, done_at, obs_log, act_log, exec_log, phase_log = 0, None, [], [], [], []
    q0 = env.d.qpos.copy()
    for t in range(max_steps):
        seen = o
        if obs_noise:
            seen = o.copy(); n = rng.normal(0, obs_noise, 3)
            seen[8:11] += n; seen[16:19] += n; seen[19:21] -= n[:2]
        if policy is None:
            a, phase = expert(seen)
        else:
            a, phase = policy(seen), None
        ex = np.array(a, float)
        if action_noise:
            ex[:3] += rng.normal(0, action_noise, 3)
        if log:
            obs_log.append(o); act_log.append(a); exec_log.append(ex)
            phase_log.append(PHASES.index(phase))
        o = env.step(ex)
        info = env.info()
        held = held + 1 if info["success"] else 0
        if held == 1:
            done_at = t + 1
        if held == 0:
            done_at = None
        if held >= HOLD_STEPS:
            break
    info = env.info()
    out = dict(success=held >= HOLD_STEPS, steps=done_at if held >= HOLD_STEPS else None,
               place_error=info["place_error"], cube_xy=env.cube_xy0.copy(), info=info)
    if log:
        out.update(obs=np.array(obs_log, np.float32), act=np.array(act_log, np.float32),
                   act_exec=np.array(exec_log, np.float64),
                   phase=np.array(phase_log, np.int8), q0=q0, final_cube=env.cube_pos())
    return out
