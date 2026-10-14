import numpy as np, matplotlib.pyplot as plt
log = np.load("panda_log.npy")
print(log.shape)                 # about (5000, 19)
t, q1 = log[:, 0], log[:, 1]     # column 1 is qpos[0] (joint 1)
plt.plot(t, 0.8*np.sin(t), "--", label="target")
plt.plot(t, q1, label="actual")
plt.legend(); plt.xlabel("time (s)"); plt.ylabel("joint1 (rad)"); plt.show()