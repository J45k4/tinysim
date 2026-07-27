"""Run 4,096 point-mass pendulums.

Parameters: mass=1 kg, length=1 m, gravity=9.81 m/s², damping=0.05,
dt=0.01 s, float32 on the selected tinygrad device. Integration is
semi-implicit Euler; contact and constraint solving are disabled. The expected
behavior is damped motion toward the downward equilibrium, with the printed
one-step control gradient agreeing with central finite differences.
"""

from tinygrad import Tensor

from tinysim.analytical import make_jitted_pendulum_step, pendulum_step


def control_gradient() -> tuple[float, float]:
  worlds, epsilon = 32, 1e-2
  theta = Tensor.full((worlds, 1), 0.2)
  omega = Tensor.zeros(worlds, 1)
  torque = Tensor.zeros(worlds, 1)
  loss = pendulum_step(theta, omega, torque)[0].mean()
  loss.backward()
  autodiff = torque.grad.sum().item()
  plus = pendulum_step(theta.detach(), omega.detach(), torque.detach() + epsilon)[0].mean().item()
  minus = pendulum_step(theta.detach(), omega.detach(), torque.detach() - epsilon)[0].mean().item()
  return autodiff, (plus - minus) / (2 * epsilon)


def main() -> None:
  worlds, steps = 4096, 1000
  theta = Tensor.full((worlds, 1), 0.2).realize()
  omega = Tensor.zeros(worlds, 1).realize()
  torque = Tensor.zeros(worlds, 1).realize()
  step = make_jitted_pendulum_step()
  for _ in range(steps):
    theta, omega = step(theta, omega, torque)
  autodiff, finite_difference = control_gradient()
  print(f"world 0 after {steps} steps: theta={theta[0, 0].item():.6f}, omega={omega[0, 0].item():.6f}")
  print(f"control gradient: autodiff={autodiff:.8e}, finite_difference={finite_difference:.8e}")


if __name__ == "__main__":
  main()
