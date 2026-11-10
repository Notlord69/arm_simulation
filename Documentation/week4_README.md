# Week 4: Panda pick-and-place, no brain

Setup (run in this folder):
    pip install mujoco numpy scikit-learn matplotlib "imageio[ffmpeg]"
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git

mujoco_menagerie must sit next to arm/. If yours is elsewhere, change MENAGERIE at the top of arm/panda_env.py.

Run from inside arm/:
    python demo.py                                  watch the expert live
    python test_controller.py ../results/controller.json ../results/controller_trace.npz
    python evaluate.py ../results                   success rate and stress tests (about 5 min)
    python collect.py ../results/pick_place_expert.npz    rebuild the dataset (about 3 min)
    python checks.py ../results/pick_place_expert.npz ../results
    python imitation_seeds.py ../results/pick_place_expert.npz ../results both

Files: panda_env.py scene, controller, success test. expert.py scripted expert. rollout.py runs one episode.
Numbers can differ slightly on another CPU, because MuJoCo is only exactly repeatable on the same build and machine.

Load the data:
    D = np.load("results/pick_place_expert.npz")
    ep = D["episode"]; train = ~D["ep_is_val"][ep] & D["ep_success"][ep]
    X, y = D["obs"][train], D["act"][train]
