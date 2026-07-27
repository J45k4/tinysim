"""Run 4,096 analytical cart-poles near the unstable upright equilibrium.

Parameters: 1 kg cart, 0.1 kg point-mass pole, 0.5 m length, gravity
9.81 m/s², dt=0.01 s, float32 on the selected tinygrad device. Integration is
semi-implicit Euler; contact and constraint solving are disabled. With zero
force and a 0.05 rad displacement, the pole is expected to fall away from the
upright configuration while every world follows the same trajectory.
"""

from tinygrad import Tensor

from tinysim.analytical import make_jitted_cartpole_step


def main() -> None:
  worlds, steps = 4096, 500
  position = Tensor.zeros(worlds, 1).realize()
  velocity = Tensor.zeros(worlds, 1).realize()
  theta = Tensor.full((worlds, 1), 0.05).realize()
  omega = Tensor.zeros(worlds, 1).realize()
  force = Tensor.zeros(worlds, 1).realize()
  step = make_jitted_cartpole_step()
  for _ in range(steps):
    position, velocity, theta, omega = step(position, velocity, theta, omega, force)
  print(f"world 0 after {steps} steps: x={position[0, 0].item():.6f}, theta={theta[0, 0].item():.6f}")


if __name__ == "__main__":
  main()
