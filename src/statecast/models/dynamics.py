"""Exact transition of the talent-plus-level Ornstein-Uhlenbeck model."""

import jax.numpy as jnp
from jax import Array

STATE_DIM = 2


def talent_transition(
    mean_reversion: Array, diffusion: Array, dt: Array
) -> tuple[Array, Array]:
    """Exact ``[talent, level]`` transition over a gap of ``dt`` days.

    Talent follows ``d theta = kappa (level - theta) dt + sigma dW`` and the
    level is constant. With ``d = exp(-kappa dt)``:

    ``Phi = [[d, 1 - d], [0, 1]]`` and
    ``Q = [[sigma^2 (1 - d^2) / (2 kappa), 0], [0, 0]]``.

    ``expm1`` keeps ``1 - d`` accurate when ``kappa dt`` is small.

    Args:
        mean_reversion: Reversion rate ``kappa`` (per day).
        diffusion: Diffusion scale ``sigma`` (metric units per root day).
        dt: Gap in days (0 gives the identity and zero noise).

    Returns:
        ``(Phi, Q)``, each shape ``(2, 2)``.

    Example:
        >>> import statecast.models
        >>> phi, q = talent_transition(jnp.asarray(0.5), jnp.asarray(2.0), 1.0)
        >>> round(float(phi[0, 0]), 4), round(float(q[0, 0]), 4)
        (0.6065, 2.5285)
    """
    decay = jnp.exp(-mean_reversion * dt)
    one_minus_decay = -jnp.expm1(-mean_reversion * dt)
    noise_var = (
        diffusion**2 * -jnp.expm1(-2.0 * mean_reversion * dt) / (2.0 * mean_reversion)
    )
    transition = jnp.array([[decay, one_minus_decay], [0.0, 1.0]])
    noise = jnp.array([[noise_var, 0.0], [0.0, 0.0]])
    return transition, noise
