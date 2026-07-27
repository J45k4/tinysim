"""Run a short same-device policy rollout for a free-base quadruped.

The model has a 5 kg trunk, four 0.5 kg legs, four torque motors, dt=0.002 s,
semi-implicit Euler, and smooth contact (stiffness 8,000, damping 150,
friction 1). The fixed constraint-iteration setting is 12 but is unused in
smooth mode. State uses float32 on the selected tinygrad device. Sixteen
randomized worlds should complete 32 policy steps with finite [16,11] qpos.
"""

from tinygrad import Tensor

from tinysim.environment import DomainRandomization, Environment
from tinysim.robots import quadruped_spec
from tinysim.simulation import Simulator
from tinysim.state import State


def main() -> None:
    simulator = Simulator.compile(quadruped_spec(), worlds=16)
    environment = Environment(
        simulator,
        randomization=DomainRandomization(
            mass_scale=(0.9, 1.1),
            friction_scale=(0.8, 1.2),
            actuator_gain_scale=(0.9, 1.1),
        ),
    )
    state, _ = environment.reset(seed=7)
    initial = state.qpos
    initial[:, 2].assign(0.55)
    state = State(
        initial,
        state.qvel,
        state.ctrl,
        state.time,
        state.constraint_impulse,
        state.parameters,
    )

    def policy(observation: Tensor) -> Tensor:
        pattern = Tensor([[1.0, -1.0, -1.0, 1.0]]).expand(16, 4)
        return observation[:, :1] * 0.0 + pattern * 0.25

    final = environment.rollout(state, policy, steps=32, inference=True)
    final.qpos.realize()
    print(
        f"worlds={simulator.worlds} device={simulator.model.device} "
        f"shape={final.qpos.shape}"
    )


if __name__ == "__main__":
    main()
