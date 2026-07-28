# TinySim

TinySim is an experimental batched robotics simulator whose numerical work is
expressed with tinygrad tensors and lowered through tinygrad's UOp compiler.
The current implementation is a deliberately small research engine, not a
MuJoCo replacement.

Implemented:

- Batched analytical pendulum and cart-pole systems.
- Static body trees with fixed, hinge, slide, ball, and free joints, separate
  `qpos`/`qvel` layouts, and depth schedules.
- Quaternion/spatial math, depth-grouped kinematics, CRBA mass matrices,
  recursive bias forces, and differentiable batched Cholesky solves.
- Motor, position, and velocity actuators plus semi-implicit integration.
- Compiler-owned fixed collision candidates for sphere, capsule, box, and
  plane primitives.
- Integrated smooth contact and frictionless constraint contact with warm
  starts, activity masks, and scalar joint limits.
- A strict native MJCF subset importer for trees, primitive geoms, defaults,
  contact pairs/exclusions, actuators, ball/free joints, and keyframes.
- Per-world mass, friction, and actuator-gain randomization, masked resets,
  same-device policy rollouts, and a representative free-base quadruped.
- Optional MuJoCo intermediate-value validation.
- A six-scenario verification matrix, saved trajectories, host-only renderers,
  release manifests/reports, PPM output, and isolated optional MP4 encoding.

## Clone and run

Tinygrad is pinned as a git submodule and its revision is also recorded in
`research-lock.json`.

```sh
git clone --recurse-submodules https://github.com/J45k4/tinysim.git
cd tinysim
```

If the repository was cloned without submodules:

```sh
git submodule update --init --recursive
```

```sh
export PYTHONPATH=.:tinygrad
export CACHEDB=/tmp/tinysim-tinygrad-cache.db
export DEV=CPU

python3 -m unittest discover -s tests -v
python3 examples/pendulum.py
python3 examples/cartpole.py
python3 examples/bouncing_ball.py
python3 -m benchmarks.bench_matrix --quick
python3 -m tinysim.verify --scenario pendulum --output artifacts/verify/pendulum
python3 examples/locomotion.py
```

The bouncing-ball example can record any selected batched world. It streams a
compact trajectory sidecar and renders the MP4 from that bounded-memory stream:

```sh
python3 examples/bouncing_ball.py --worlds 256 --steps 2000 \
  --record artifacts/bouncing-ball-example.mp4
```

To create an MP4, install FFmpeg or GStreamer/OpenH264 and add
`--record artifacts/pendulum.mp4` to the verification command. Without FFmpeg,
verification can use the isolated GStreamer adapter; without either encoder it
still writes the numerical trajectory, manifest, and deterministic first/last
PPM frames. `--scenario all` is the strict release command: it writes
`release-report.json` and exits with status 2 until every evidence, accelerator
recording, Critic, and manager-review gate is satisfied.

The accelerator evidence run records the batched-world scenario directly:

```sh
DEV=CUDA python3 -m tinysim.verify --scenario all \
  --output artifacts/verify \
  --accelerator-record artifacts/verify/accelerator-batched.mp4
```

After the manager signs the hashes of the golden and accelerator videos, refresh
only the release index. This never reruns physics or re-encodes either file:

```sh
python3 -m tinysim.verify --refresh-release --output artifacts/verify
```

## Public simulator

```python
from tinysim import ActuatorSpec, BodySpec, JointSpec, ModelSpec, Simulator

model = ModelSpec(
    bodies=[BodySpec("bob", mass=1.0, com=(0.0, 0.0, -1.0))],
    joints=[JointSpec("hinge", body=0, kind="hinge", axis=(0.0, 1.0, 0.0))],
    actuators=[ActuatorSpec("motor", joint="hinge")],
    timestep=0.01,
)
simulator = Simulator.compile(model, worlds=256)
state = simulator.make_state()
control = simulator.zeros_control()
state = simulator.step(state, control)
observation = simulator.observe(state)
```

`Simulator.step` is the differentiable eager path. For forward-only rollout,
`Simulator.inference_step` uses `TinyJit`; its outputs are deliberately detached
because TinyJit replay does not preserve an autograd graph across the captured
call boundary. Focused analytical/reference JIT helpers have the same
forward-only contract but are not the public rollout API.

For a stateful rollout, use `Simulation`. It compiles the model, owns the
current state, and only performs host reads when recording is requested:

```python
from tinysim import Simulation

simulation = Simulation(model, worlds=256)
simulation.reset(qpos=initial_qpos)
simulation.run(steps=2000)

# Equivalently: simulation.record("rollout.mp4", steps=2000, world=0)
simulation.run(
    steps=2000,
    record="rollout.mp4",
    world=0,
    fps=60,
)

# Tile selected batched worlds into one 4x4 video.
simulation.record(
    "rollout-grid.mp4",
    steps=2000,
    worlds=range(16),
    columns=4,
)
```

Recorded runs save a versioned `.trajectory.tstraj` sidecar and render supported
sphere, capsule, box, and plane geometry through the host-side primitive
renderer. Samples are packed into one compact device-to-host transfer and
written immediately, so host memory does not grow with trajectory length. Grid
recordings copy and store only the selected worlds. Initialize those worlds
with different states or parameters to compare their rollouts.
`Simulation.from_compiled(...)` is available for advanced checkpoint or
compilation-reuse workflows. Use
`tinysim.trajectory.export_trajectory_json(...)` when a portable JSON copy is
needed.

The Jenga stress example exercises oriented box-box SAT collision, free-body
rotation, friction, many fixed collision pairs, TinyJit, and grid recording:

```bash
PYTHONPATH=.:tinygrad DEV=CUDA python3 examples/jenga.py \
  --levels 10 --worlds 16 --steps 5000 \
  --record artifacts/jenga-manifolds-orbit-16.mp4 \
  --record-grid 16 --grid-columns 4 --record-every 10 \
  --fps 30 --camera-turns 1 --camera-elevation 20
```

This is deliberately a collapse test rather than a resting-stack benchmark.
TinySim uses fixed four-slot box manifolds, with inactive slots masked for
edge/vertex contacts and smooth load distributed over active points. Tall
towers default to a declared support graph (corresponding adjacent-level blocks
plus every block against the floor). `--pair-mode local` adds all adjacent and
same-level pairs; `--pair-mode all` compiles the quadratic full graph.
The command records a five-second, 4-by-4 grid while the orthographic 3D camera
completes one orbit. `Simulation.run(...)` and `Simulation.record(...)` also
accept a fixed `Camera3D` or a reusable `OrbitCamera` path.

MJCF loading is available either as a diagnostic-rich imported packet or
directly as a `ModelSpec`:

```python
from tinysim import ModelSpec, Simulator, load_mjcf

imported = load_mjcf("robot.xml")
simulator = Simulator.compile(imported.spec, worlds=256)

spec = ModelSpec.from_mjcf("robot.xml")
simulator = Simulator.compile(
    spec,
    worlds=256,
    contact="smooth",
    solver_iterations=8,
)
```

## Conventions and current limits

Runtime state uses an explicit leading world dimension. Coordinates are
right-handed SI; quaternions are `(w, x, y, z)`; collision distance is positive
when separated; contact normals point from shape A toward shape B.

The articulated path requires parent-before-child trees, one joint per body,
diagonal body-frame inertia, and explicit inertials. Actuators on ball/free
joints require an explicit transmission and are rejected for now. Mesh
collision, constraint friction, equality constraints, general external
wrenches, stateful actuators, and URDF import are not silently approximated.
