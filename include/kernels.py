import jax.numpy as jnp
from jax import jit


@jit
def rbf_kernel(x1, x2, initial_theta):
    """Computes the Gaussian RBF ARD kernel for inputs x1 and x2.

    Args:
        x1: An array of shape (N1, D) where N1 is the number of data points in x1
            and D is the number of dimensions.
        x2: An array of shape (N2, D) where N2 is the number of data points in x2
            and D is the number of dimensions.
        length_scale: A scalar or array of shape (D,) representing the length scale
            for each dimension.
        sigma_f: A scalar representing the standard deviation of the Gaussian RBF.

    Returns:
        A matrix of shape (N1, N2) representing the Gaussian RBF ARD kernel
        between x1 and x2.
    """
    square_distances = jnp.sum((x1[:, jnp.newaxis, :] - x2[jnp.newaxis, :, :]) ** 2 / initial_theta['lengthscale'] ** 2,
                               axis=-1)
    return initial_theta['sigma'] ** 2 * jnp.exp(-0.5 * square_distances)


def matern_kernel_3_2(x1, x2, initial_theta):
    assert nu in [0.5, 1.5, 2.5], "nu must be 0.5, 1.5, or 2.5"

    lengthscale, sigma = initial_theta['lengthscale'], initial_theta['sigma']

    square_distances = jnp.sum((x1[:, jnp.newaxis, :] - x2[jnp.newaxis, :, :]) ** 2 / lengthscale ** 2, axis=-1)
    distances = jnp.sqrt(square_distances)

    kernel = (1 + jnp.sqrt(3) * distances) * jnp.exp(-jnp.sqrt(3) * distances)

    return sigma ** 2 * kernel


@jit
def matern_kernel_5_2(x1, x2, initial_theta):
    assert nu in [0.5, 1.5, 2.5], "nu must be 0.5, 1.5, or 2.5"

    lengthscale, sigma = initial_theta['lengthscale'], initial_theta['sigma']

    square_distances = jnp.sum((x1[:, jnp.newaxis, :] - x2[jnp.newaxis, :, :]) ** 2 / lengthscale ** 2, axis=-1)
    distances = jnp.sqrt(square_distances)

    kernel = (1 + jnp.sqrt(5) * distances + 5 / 3 * square_distances) * jnp.exp(-jnp.sqrt(5) * distances)

    return sigma ** 2 * kernel


@jit
def rational_quadratic_kernel(x1, x2, initial_theta):
    lengthscale, alpha, sigma = initial_theta['lengthscale'], initial_theta['alpha'], initial_theta['sigma']

    square_distances = jnp.sum((x1[:, jnp.newaxis, :] - x2[jnp.newaxis, :, :]) ** 2 / lengthscale ** 2, axis=-1)

    return sigma ** 2 * (1 + square_distances / (2 * alpha)) ** (-alpha)


@jit
def spectral_mixture_kernel(x1, x2, initial_theta):
    weights, length_scales, mu = initial_theta['weights'], initial_theta['length_scales'], initial_theta['mu']

    x1_exp = x1[:, jnp.newaxis, jnp.newaxis, :]
    x2_exp = x2[jnp.newaxis, jnp.newaxis, :, :]
    mu = mu[jnp.newaxis, :, jnp.newaxis, :]

    diff = x1_exp - x2_exp
    scaled_square_dist = (diff ** 2) / length_scales ** 2
    exponent_term = jnp.exp(-0.5 * jnp.sum(scaled_square_dist, axis=-1))

    cos_term = jnp.cos(2 * jnp.pi * jnp.sum(diff * mu, axis=-1))

    return jnp.sum(weights * exponent_term * cos_term, axis=-2)
