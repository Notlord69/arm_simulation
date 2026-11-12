"""Week 8: an expert that is easier to imitate at the last step.

Week 7's failure: the week 4 expert lowers the cube only once it is within 8 mm of the target (a
sharp cut-off in a distance the brain can't resolve that finely), so near the target its dz label
is 0 almost everywhere and the brain-driven arm hovered 20-30 mm away holding the cube until time
ran out.

The graded expert changes only the last two stages. Everything else (reach, grasp, lift, carry,
retreat, re-grasp) is week 4's expert unchanged, and it is still memoryless (a pure function of
obs), so DAgger can query it anywhere. The success test is unchanged (5 cm zone).

1. Graded lowering. While carrying, the gripper's height target slides from travel height to
   placing height as the cube approaches the target,
       h = clip((d_goal - FULL_LOWER) / (START_LOWER - FULL_LOWER), 0, 1)
       z_target = z_place + h * (HOVER_Z - z_place)
   while xy keeps correcting toward the target: dz is negative and grows smoothly over a band
   instead of switching on at 8 mm. The band ends at 25 mm (v3): the zone allows 50 mm, and in the
   sandbox a readout whose xy settled ~2 cm off target hovered half-lowered when the band ended at
   10 mm (v2). The expert itself still places within ~3 mm, because xy keeps correcting.
2. Press down, then let go (v2, after the sandbox showed the next failure). Week 4 lowers toward a
   point 3 mm above resting height and opens only once the cube is within 6 mm of the table. Its
   proportional dz shrinks to a few mm/step in the last centimetre, and the stand-in brain stalled
   there, cube gripped 1-2 cm up. Now the placing target is PRESS below resting height (the table
   stops the cube, so dz stays clearly negative until contact), and the gripper opens once the cube
   is within RELEASE_DZ of the table and within RELEASE_XY of the target.
"""
import numpy as np

from expert import (expert as expert4, move, HOVER_Z, CUBE_HALF, STRADDLE_XY, HELD_W, OPEN, CLOSE)

START_LOWER = 0.050   # m, cube-to-target distance where lowering starts
FULL_LOWER = 0.025    # m, distance from which the height target is the placing height
LIFT_FIRST = 0.045    # m, farther than this and low: lift vertically first (as week 4)
PRESS = 0.008         # m, placing target this far below the cube's resting height (v2)
RELEASE_DZ = 0.010    # m, open once the cube is this close to resting height (week 4: 0.006)
RELEASE_XY = 0.040    # m, ... and this close to the target (week 4: 0.030; zone radius 0.050)


def make_expert8(start=START_LOWER, full=FULL_LOWER, press=PRESS, release_dz=RELEASE_DZ, release_xy=RELEASE_XY):
    def expert8(o):
        tcp, width, wvel = o[:3], o[6], o[7]
        cube, goal = o[8:11], o[14:16]
        rel = cube - tcp
        rel_xy = np.linalg.norm(rel[:2])
        d_goal = np.linalg.norm(cube[:2] - goal)
        straddled = rel_xy < STRADDLE_XY and abs(rel[2]) < 0.025
        gripping = straddled and width < HELD_W and (abs(wvel) < 0.05 or cube[2] > CUBE_HALF + 0.01)

        # cube low and at the target: let go, then leave (week 4's rule, wider thresholds)
        if d_goal < release_xy and cube[2] < CUBE_HALF + release_dz:
            if width < CUBE_HALF * 2 + 0.02 and straddled:
                return np.r_[0, 0, 0, OPEN], "release"
            return np.r_[move(tcp, np.r_[tcp[:2], HOVER_Z]), OPEN], "retreat"
        if not gripping:
            return expert4(o)                      # reach, descend, close: week 4 unchanged
        z_place = CUBE_HALF - press - rel[2]
        if d_goal > max(start, LIFT_FIRST) and tcp[2] < HOVER_Z - 0.03:
            return np.r_[move(tcp, np.r_[tcp[:2], HOVER_Z]), CLOSE], "lift"
        h = float(np.clip((d_goal - full) / (start - full), 0, 1))
        z_t = z_place + h * (HOVER_Z - z_place)
        return np.r_[move(tcp, np.r_[goal - rel[:2], z_t]), CLOSE], ("carry" if h >= 1 else "lower")
    return expert8


expert8 = make_expert8()                                                        # v3: wide ramp + press down
expert8_v1 = make_expert8(start=0.040, full=0.010, press=-0.003, release_dz=0.006, release_xy=0.030)  # graded only
expert8_v2 = make_expert8(start=0.040, full=0.010, release_xy=0.035)            # + press down
