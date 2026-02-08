import json, math, os

# ----------  customise your force laws  -------------
def force_phase1(pos):
    r = math.sqrt(sum(c*c for c in pos))
    return [0,0,0] if r == 0 else [-c / r**3 for c in pos]

def force_phase2(pos):
    r = math.sqrt(sum(c*c for c in pos))
    return [0,0,0] if r == 0 else [-0.5 * c / r**3 for c in pos]
# ----------------------------------------------------

START, END, FPS = 1, 250, 20
dt = 1 / FPS
# replace with your real initial positions
seed = [[0.95,0,0], [0,0.95,0], [-0.95,0,0]]

def simulate(force):
    state = [p[:] for p in seed]
    ref = {}
    for f in range(START, END+1):
        ref[str(f)] = [p[:] for p in state]
        for i,p in enumerate(state):
            F = force(p)
            state[i] = [p[j]+F[j]*dt for j in range(3)]
    return ref

with open("phase1_reference.json", "w") as f:
    json.dump(simulate(force_phase1), f, indent=2)
with open("phase2_reference.json", "w") as f:
    json.dump(simulate(force_phase2), f, indent=2)

print("✅  Wrote phase1_reference.json and phase2_reference.json")
