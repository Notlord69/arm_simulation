"""Week 6 video: the arm on the left, the fly brain's descending neurons on the right.

Run as a script so the OpenGL backend can be chosen before MuJoCo is imported:

    python -m sim.video <episodes.pkl> <out.mp4> [--slow 0.5] [--gl egl]

episodes.pkl: list of dicts saved by the week 6 notebook, one per episode:
    rec      list of per-step dicts from run_episode(record=True)
    seed, success, knocked, steps
    dn_order optional: row order for the raster (e.g. sorted by readout weight)

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
    from panda_env import PandaPickPlace

    eps = pickle.load(open(pkl, 'rb'))
    env = PandaPickPlace()
    ren = mujoco.Renderer(env.m, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.47, -0.05, 0.16]                  # side view: gripper, cube and table edge
    cam.distance, cam.azimuth, cam.elevation = 1.3, 90, -16
    ink, ink2, blue, orange, aqua = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834', '#1baf7a'
    box = dict(facecolor='black', alpha=0.55, edgecolor='none', pad=4)
    fps = fps_ctrl * slow
    writer = imageio.get_writer(out, fps=fps, codec='libx264', quality=8, macro_block_size=8)
    WIN_S = 1.0                                     # raster shows the last second of brain time

    for e_i, ep in enumerate(eps):
        rec = ep['rec']
        steps = [r for r in rec if r['act'] is not None]
        n_dn = steps[0]['counts'].shape[1] if steps and steps[0]['counts'] is not None else 0
        order = np.asarray(ep.get('dn_order', np.arange(n_dn)))
        # spike times (s) per DN, built from 5 ms bins: one marker per spike count in a bin
        bins = np.concatenate([r['counts'] for r in steps]) if n_dn else np.zeros((0, 0))
        act = np.array([r['act'] for r in steps]) * 1000      # mm per step
        lab = np.array([r['label'] for r in steps]) * 1000
        t_ctrl = (np.arange(len(steps)) + 1) * 0.04
        verdict = 'REACHED' if ep['success'] else ('knocked the cube' if ep['knocked'] else 'timed out')
        for t, r in enumerate(rec):
            env.d.qpos[:] = r['qpos']
            mujoco.mj_forward(env.m, env.d)
            ren.update_scene(env.d, cam)
            img = ren.render()
            fig = plt.figure(figsize=(12.8, 4.8), dpi=100)
            ax0 = fig.add_axes([0, 0, 0.5, 1]); ax0.imshow(img); ax0.axis('off')
            tcp, cube = env.tcp_pos(), env.cube_pos()
            dxy = 1000 * np.hypot(*(cube - tcp)[:2])
            ax0.text(12, 26, f"episode {e_i + 1}/{len(eps)}  (seed {ep['seed']})", color='white',
                     fontsize=11, weight='bold', bbox=box)
            ax0.text(12, 56, f"t = {t * 0.04:.2f} s   gripper-cube xy {dxy:.0f} mm   height {1000 * tcp[2]:.0f} mm",
                     color='white', fontsize=9.5, bbox=box)
            if t == len(rec) - 1:
                ax0.text(w / 2, h - 30, verdict, color='#6ee7a0' if ep['success'] else '#ffb38a',
                         fontsize=18, weight='bold', ha='center', bbox=box)
            # raster
            ax1 = fig.add_axes([0.56, 0.5, 0.42, 0.42])
            now = min(t, len(steps)) * 0.04
            if n_dn:
                upto = min(t, len(steps)) * 8
                b, n = np.nonzero(bins[:upto][:, order]) if upto else (np.array([], int), np.array([], int))
                ax1.scatter(b * 0.005 + 0.0025, n, s=2, marker='|', c=ink, lw=0.8)
                ax1.set_ylim(-1, n_dn)
            ax1.set_xlim(max(0, now - WIN_S), max(WIN_S, now))
            ax1.set_ylabel('descending neurons', color=ink2, fontsize=9)
            ax1.set_title('fly brain: DN-candidate spikes (brain is driving)', fontsize=10,
                          color=ink, loc='left', weight='bold')
            ax1.set_xticklabels([]); ax1.tick_params(colors=ink2, labelsize=8)
            for s in ('top', 'right'): ax1.spines[s].set_visible(False)
            # action trace
            ax2 = fig.add_axes([0.56, 0.1, 0.42, 0.28])
            k = min(t, len(steps))
            for j, (nm, cl) in enumerate((('dx', blue), ('dy', orange), ('dz', aqua))):
                ax2.plot(t_ctrl[:k], act[:k, j], color=cl, lw=1.8, label=nm)
                ax2.plot(t_ctrl[:k], lab[:k, j], color=cl, lw=1, ls=':', alpha=0.9)
            ax2.set_xlim(max(0, now - WIN_S), max(WIN_S, now)); ax2.set_ylim(-24, 24)
            ax2.set_xlabel('time (s)', color=ink2, fontsize=9)
            ax2.set_ylabel('mm per step', color=ink2, fontsize=9)
            ax2.tick_params(colors=ink2, labelsize=8)
            ax2.legend(fontsize=7.5, ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False)
            ax2.text(1.0, 1.04, 'solid: decoded from the brain   dotted: expert, for comparison only',
                     transform=ax2.transAxes, ha='right', va='bottom', fontsize=7, color=ink2)
            for s in ('top', 'right'): ax2.spines[s].set_visible(False)
            ax2.axhline(0, color='#e6e5e1', lw=0.8, zorder=0)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            plt.close(fig)
            reps = int(round(fps * 0.8)) if t == len(rec) - 1 else 1     # hold the last frame
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
        p = subprocess.run([sys.executable, '-m', 'sim.video', pkl, out, '--slow', str(slow), '--gl', b,
                            '--child'], env=env, cwd=os.path.dirname(os.path.dirname(__file__)),
                           capture_output=True, text=True)
        if p.returncode == 0:
            print(p.stdout.strip(), f"(OpenGL: {b})")
            return b
        print(f"  {b}: failed ({(p.stderr.strip().splitlines() or ['?'])[-1][:120]})")
    print("No OpenGL backend worked here. Copy the .pkl to Colab and run: "
          "!MUJOCO_GL=egl python -m sim.video episodes.pkl reach.mp4")
    return None


if __name__ == '__main__':
    a = sys.argv[1:]
    slow = float(a[a.index('--slow') + 1]) if '--slow' in a else 0.5
    gl = a[a.index('--gl') + 1] if '--gl' in a else None
    if '--child' in a:
        _render(a[0], a[1], slow)
    else:
        render(a[0], a[1], slow, gl)
