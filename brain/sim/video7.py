"""Week 7 video: the arm on the left; on the right the readout neurons' spikes and the decoded
action (dx, dy, dz, grip) against what the expert would have done.

Run as a script so the OpenGL backend can be chosen before MuJoCo is imported:

    python -m sim.video7 <episodes.pkl> <out.mp4> [--slow 0.5] [--gl egl]

episodes.pkl: list of dicts saved by the week 7 notebook, one per episode:
    rec      list of per-step dicts from run_pp(record=True)
    seed, success, outcome, net (e.g. 'connectome', 'random network', 'MLP')

With no --gl given it tries egl, then osmesa, then glfw, each in its own process.
"""
import os, sys, subprocess, pickle

ARM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'arm')


def _render(pkl, out, slow, fps_ctrl=25, w=640, h=480):
    import numpy as np
    import mujoco
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    sys.path.insert(0, ARM)
    from panda_env import PandaPickPlace, TARGET_XY

    eps = pickle.load(open(pkl, 'rb'))
    env = PandaPickPlace()
    ren = mujoco.Renderer(env.m, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.48, 0.03, 0.12]                  # sees the spawn area and the target zone
    cam.distance, cam.azimuth, cam.elevation = 1.45, 145, -24
    ink, ink2, blue, orange, aqua, violet = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
    box = dict(facecolor='black', alpha=0.55, edgecolor='none', pad=4)
    fps = fps_ctrl * slow
    writer = imageio.get_writer(out, fps=fps, codec='libx264', quality=8, macro_block_size=8)
    WIN_S = 1.6

    for e_i, ep in enumerate(eps):
        rec = ep['rec']
        steps = [r for r in rec if r['act'] is not None]
        has_spk = bool(steps) and steps[0]['counts'] is not None
        n_dn = steps[0]['counts'].shape[1] if has_spk else 0
        bins = np.concatenate([r['counts'] for r in steps]) if has_spk else None
        bins_per_step = steps[0]['counts'].shape[0] if has_spk else 8
        act = np.array([r['act'] for r in steps])
        lab = np.array([r['label'] for r in steps])
        t_ctrl = (np.arange(len(steps)) + 1) * 0.04
        verdict = 'PLACED' if ep['success'] else ep['outcome'].upper()
        for t, r in enumerate(rec):
            env.d.qpos[:] = r['qpos']
            mujoco.mj_forward(env.m, env.d)
            ren.update_scene(env.d, cam)
            img = ren.render()
            fig = plt.figure(figsize=(12.8, 4.8), dpi=100)
            ax0 = fig.add_axes([0, 0, 0.5, 1]); ax0.imshow(img); ax0.axis('off')
            tcp, cube = env.tcp_pos(), env.cube_pos()
            ax0.text(12, 26, f"{ep['net']}  -  episode {e_i + 1}/{len(eps)} (seed {ep['seed']})", color='white',
                     fontsize=11, weight='bold', bbox=box)
            phase = f"   (expert would: {r['phase']})" if r['phase'] else ''
            ax0.text(12, 56, f"t = {t * 0.04:.2f} s   cube to target {1000 * np.hypot(*(cube[:2] - TARGET_XY)):.0f} mm"
                     f"   cube height {1000 * cube[2]:.0f} mm{phase}",
                     color='white', fontsize=9, bbox=box)
            if t == len(rec) - 1:
                ax0.text(w / 2, h - 30, verdict, color='#6ee7a0' if ep['success'] else '#ffb38a',
                         fontsize=18, weight='bold', ha='center', bbox=box)
            now = min(t, len(steps)) * 0.04
            ax1 = fig.add_axes([0.56, 0.52, 0.42, 0.40])
            if has_spk:
                upto = min(t, len(steps)) * bins_per_step
                b, n = np.nonzero(bins[:upto]) if upto else (np.array([], int), np.array([], int))
                ax1.scatter(b * 0.005 * 8 / bins_per_step + 0.0025, n, s=2, marker='|', c=ink, lw=0.8)
                ax1.set_ylim(-1, n_dn)
                ax1.set_title(f"{ep['net']}: readout-neuron spikes (the network is driving)", fontsize=10,
                              color=ink, loc='left', weight='bold')
                ax1.set_ylabel('readout neurons', color=ink2, fontsize=9)
            else:
                ax1.text(0.5, 0.5, 'no spiking network in this policy', ha='center', va='center',
                         color=ink2, transform=ax1.transAxes)
                ax1.set_title(f"{ep['net']}", fontsize=10, color=ink, loc='left', weight='bold')
            ax1.set_xlim(max(0, now - WIN_S), max(WIN_S, now))
            ax1.set_xticklabels([]); ax1.tick_params(colors=ink2, labelsize=8)
            for s in ('top', 'right'): ax1.spines[s].set_visible(False)
            ax2 = fig.add_axes([0.56, 0.1, 0.42, 0.30])
            k = min(t, len(steps))
            for j, (nm, cl) in enumerate((('dx', blue), ('dy', orange), ('dz', aqua))):
                ax2.plot(t_ctrl[:k], act[:k, j] * 1000, color=cl, lw=1.8, label=nm)
                ax2.plot(t_ctrl[:k], lab[:k, j] * 1000, color=cl, lw=1, ls=':', alpha=0.9)
            ax2.plot(t_ctrl[:k], np.clip(act[:k, 3], -1.5, 1.5) * 15, color=violet, lw=1.8, label='grip (+ open)')
            ax2.plot(t_ctrl[:k], lab[:k, 3] * 15, color=violet, lw=1, ls=':')
            ax2.set_xlim(max(0, now - WIN_S), max(WIN_S, now)); ax2.set_ylim(-26, 26)
            ax2.set_xlabel('time (s)', color=ink2, fontsize=9)
            ax2.set_ylabel('mm per step', color=ink2, fontsize=9)
            ax2.tick_params(colors=ink2, labelsize=8)
            ax2.legend(fontsize=7.5, ncol=4, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False)
            ax2.text(1.0, 1.04, 'solid: decoded   dotted: expert, for comparison only',
                     transform=ax2.transAxes, ha='right', va='bottom', fontsize=7, color=ink2)
            for s in ('top', 'right'): ax2.spines[s].set_visible(False)
            ax2.axhline(0, color='#e6e5e1', lw=0.8, zorder=0)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            plt.close(fig)
            reps = int(round(fps * 0.8)) if t == len(rec) - 1 else 1
            for _ in range(reps):
                writer.append_data(frame)
    writer.close()
    print(f"wrote {out}")


def render(pkl, out, slow=0.5, gl=None):
    """Try OpenGL backends in turn (each in a fresh process). Returns the backend used."""
    for b in ([gl] if gl else ['egl', 'osmesa', 'glfw']):
        env = dict(os.environ, MUJOCO_GL=b)
        if b == 'osmesa':
            env['PYOPENGL_PLATFORM'] = 'osmesa'
        p = subprocess.run([sys.executable, '-m', 'sim.video7', pkl, out, '--slow', str(slow), '--gl', b,
                            '--child'], env=env, cwd=os.path.dirname(os.path.dirname(__file__)),
                           capture_output=True, text=True)
        if p.returncode == 0:
            print(p.stdout.strip(), f"(OpenGL: {b})")
            return b
        print(f"  {b}: failed ({(p.stderr.strip().splitlines() or ['?'])[-1][:120]})")
    print("No OpenGL backend worked here. Copy the .pkl to Colab and run: "
          "!MUJOCO_GL=egl python -m sim.video7 week7_episodes.pkl week7.mp4")
    return None


if __name__ == '__main__':
    a = sys.argv[1:]
    slow = float(a[a.index('--slow') + 1]) if '--slow' in a else 0.5
    gl = a[a.index('--gl') + 1] if '--gl' in a else None
    if '--child' in a:
        _render(a[0], a[1], slow)
    else:
        render(a[0], a[1], slow, gl)
