"""Small vector operations used by TinySim.

Vectors are stored in the final dimension and transforms act on column
vectors.  The functions are deliberately Tensor-only so gradients and device
placement are preserved.
"""

from tinygrad import Tensor


def dot(a: Tensor, b: Tensor) -> Tensor:
    if a.shape[-1] != b.shape[-1]:
        raise ValueError("dot operands must have the same final dimension")
    return (a * b).sum(axis=-1)


def cross(a: Tensor, b: Tensor) -> Tensor:
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("cross expects 3-vectors")
    return Tensor.stack(
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        dim=-1,
    )


def norm(vector: Tensor, epsilon: float = 0.0) -> Tensor:
    return (dot(vector, vector) + epsilon).sqrt()


def normalize(vector: Tensor, epsilon: float = 1e-12) -> Tensor:
    length = norm(vector)
    safe_length = (length > epsilon).where(length, 1.0)
    return vector / safe_length.unsqueeze(-1)


def skew(vector: Tensor) -> Tensor:
    """Returns the matrix ``S`` such that ``S @ x == cross(vector, x)``."""
    if vector.shape[-1] != 3:
        raise ValueError("skew expects 3-vectors")
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    zero = x * 0.0
    return Tensor.stack(
        Tensor.stack(zero, -z, y, dim=-1),
        Tensor.stack(z, zero, -x, dim=-1),
        Tensor.stack(-y, x, zero, dim=-1),
        dim=-2,
    )


def matvec(matrix: Tensor, vector: Tensor) -> Tensor:
    if matrix.shape[-2:] != (3, 3) or vector.shape[-1] != 3:
        raise ValueError("matvec expects [..., 3, 3] and [..., 3]")
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)
