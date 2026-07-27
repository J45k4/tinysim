"""Small, fixed-iteration linear solvers expressed with tinygrad tensors."""

from tinygrad import Tensor


def _check_system(matrix: Tensor, rhs: Tensor) -> None:
    if matrix.ndim != 3 or rhs.ndim != 2:
        raise ValueError("expected matrix [world, n, n] and rhs [world, n]")
    if matrix.shape[0] != rhs.shape[0] or matrix.shape[1] != matrix.shape[2]:
        raise ValueError("matrix must be square and share rhs world dimension")
    if matrix.shape[2] != rhs.shape[1]:
        raise ValueError("matrix and rhs equation dimensions differ")


def projected_jacobi(
    matrix: Tensor,
    rhs: Tensor,
    *,
    iterations: int,
    lower: float | Tensor = 0.0,
    upper: float | Tensor | None = None,
    initial: Tensor | None = None,
    epsilon: float = 1e-9,
) -> Tensor:
    """Fixed-iteration projected Jacobi for small batched constraint systems."""
    _check_system(matrix, rhs)
    if iterations < 1:
        raise ValueError("iterations must be positive")

    diagonal = matrix.diagonal(dim1=-2, dim2=-1)
    safe_diagonal = (diagonal.abs() > epsilon).where(diagonal, epsilon)
    identity = Tensor.eye(matrix.shape[-1], dtype=matrix.dtype).to(matrix.device)
    off_diagonal = matrix - identity.unsqueeze(0) * diagonal.unsqueeze(-1)
    if initial is not None:
        if initial.shape != rhs.shape:
            raise ValueError("initial solution must match rhs shape")
        if initial.dtype != rhs.dtype or initial.device != rhs.device:
            raise TypeError("initial solution must match rhs dtype/device")
        solution = initial.maximum(lower)
        if upper is not None:
            solution = solution.minimum(upper)
    else:
        solution = rhs * 0.0

    for _ in range(iterations):
        coupled = (off_diagonal @ solution.unsqueeze(-1)).squeeze(-1)
        solution = ((rhs - coupled) / safe_diagonal).maximum(lower)
        if upper is not None:
            solution = solution.minimum(upper)

    return solution
