import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.scipy.linalg import expm

import statecast.models  # noqa: F401  (enables 64-bit floats)
from statecast.models.dynamics import talent_transition


def van_loan(drift: Array, diffusion: Array, dt: Array) -> tuple[Array, Array]:
    """Independent oracle: exact discretization of dx = F x dt + dW.

    Exponentiate [[-F, Qc], [0, F^T]] dt; Phi = B22^T and Q = Phi B12.
    """
    dim = drift.shape[0]
    block = jnp.block([[-drift, diffusion], [jnp.zeros((dim, dim)), drift.T]]) * dt
    exponential = expm(block)
    transition = exponential[dim:, dim:].T
    return transition, transition @ exponential[:dim, dim:]


@pytest.mark.parametrize("dt", [0.5, 5.0, 150.0])
def test_closed_form_matches_van_loan(dt: float) -> None:
    kappa, sigma = 0.03, 0.2
    drift = jnp.array([[-kappa, kappa], [0.0, 0.0]])
    diffusion = jnp.array([[sigma**2, 0.0], [0.0, 0.0]])

    transition, noise = talent_transition(
        jnp.asarray(kappa), jnp.asarray(sigma), jnp.asarray(dt)
    )

    expected_transition, expected_noise = van_loan(drift, diffusion, jnp.asarray(dt))
    np.testing.assert_allclose(transition, expected_transition, atol=1e-12)
    np.testing.assert_allclose(noise, expected_noise, atol=1e-12)


def test_tiny_rate_stays_accurate() -> None:
    kappa, sigma, dt = 1e-9, 0.1, 3.0

    _, noise = talent_transition(
        jnp.asarray(kappa), jnp.asarray(sigma), jnp.asarray(dt)
    )

    np.testing.assert_allclose(noise[0, 0], sigma**2 * dt, rtol=1e-6)


def test_zero_gap_is_identity() -> None:
    transition, noise = talent_transition(
        jnp.asarray(0.02), jnp.asarray(0.1), jnp.asarray(0.0)
    )

    np.testing.assert_allclose(transition, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(noise, np.zeros((2, 2)), atol=1e-12)
