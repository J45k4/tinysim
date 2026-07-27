"""Compile and step three MJCF examples.

Each XML records its masses, inertias, dt=0.01 s, gravity, primitive geoms,
joint limits, and actuators. TinySim uses float32 semi-implicit Euler with
contact mode ``none`` and the default eight constraint iterations (inactive).
Four worlds per model should advance exactly one step and print their nq/nv/nu
dimensions without requiring MuJoCo at runtime.
"""

from pathlib import Path

from tinysim import Simulator, load_mjcf


MODELS = ("pendulum.xml", "cartpole.xml", "two_link.xml")


def main() -> None:
  model_directory = Path(__file__).parent / "verification"
  for filename in MODELS:
    imported = load_mjcf(model_directory / filename)
    simulator = Simulator.compile(imported.spec, worlds=4)
    state = simulator.step(simulator.make_state(), simulator.zeros_control())
    print(
      f"{imported.spec.name}: worlds=4 nq={simulator.model.nq} "
      f"nv={simulator.model.nv} time={float(state.time[0].item()):.3f}"
    )


if __name__ == "__main__":
  main()
