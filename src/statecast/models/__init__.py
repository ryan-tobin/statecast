"""State-space models of latent player talent.

Importing this package switches JAX to 64-bit floats: Kalman covariances lose
positive-definiteness in float32 over long sequences.
"""

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
