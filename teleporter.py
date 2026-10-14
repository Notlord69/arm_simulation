import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
d = mujoco.MjData(m)

low  = m.jnt_range[:7, 0]
print(low)
