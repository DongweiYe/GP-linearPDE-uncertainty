import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.typing import ArrayLike

from include.heat2d import deep_rbf_kernel


def f_rd(Xf):
    x = Xf[:, :1]
    t = Xf[:, -1:]
    term1 = jnp.exp(t)
    term2 = jnp.sin(4 * np.pi * x)
    term3 = 16 * np.pi ** 2
    term4 = jnp.exp(t) * jnp.sin(4 * np.pi * x)
    term5 = 10 * x * jnp.exp(t)
    term6 = 25 * x ** 2 * jnp.exp(2 * t)

    return term1 * term2 * (term3 + term4 + term5) + term6


def u_rd(Xu_fixed):
    x = Xu_fixed[:, :1]
    t = Xu_fixed[:, -1:]
    return jnp.exp(t) * (jnp.sin(4 * np.pi * x) + 5 * x)


@jit
def react_diffs_high_d_kuu(x1: ArrayLike, x2: ArrayLike, params) -> jnp.ndarray:
    return deep_rbf_kernel(x1, x2, params)


@jit
def react_diffs_high_d_kff(x1: ArrayLike, x2: ArrayLike, initial_theta) -> jnp.ndarray:
    total_sum_second_order_x1_x_kuu = jnp.zeros_like(deep_second_order_x_x_kuu(x1, x2, initial_theta, 0))
    total_sum_second_order_x2_x_kuu = jnp.zeros_like(deep_second_order_x_x_kuu(x1, x2, initial_theta, 0))
    num_x1 = x1.shape[1] - 1
    num_x2 = x2.shape[1] - 1

    for d in range(num_x1):
        total_sum_second_order_x1_x_kuu += deep_second_order_x_x_kuu(x1, x2, initial_theta, d)
    for d in range(num_x2):
        total_sum_second_order_x2_x_kuu += deep_second_order_x_x_kuu(x1, x2, initial_theta, d)

    first_order_term_x1 = deep_first_order_x_t_kuu(x1, x2, initial_theta, num_x1)
    first_order_term_x2 = deep_first_order_x_t_kuu(x2, x1, initial_theta, num_x2)
    kuu_term = react_diffs_high_d_kuu(x1, x2, initial_theta)
    nonlinear_term = kuu_term ** 2
    linear_term = kuu_term

    return (first_order_term_x1 - total_sum_second_order_x1_x_kuu + nonlinear_term - linear_term) * \
        (first_order_term_x2 - total_sum_second_order_x2_x_kuu + nonlinear_term - linear_term)


@jit
def react_diffs_high_d_kfu(x1: ArrayLike, x2: ArrayLike, initial_theta) -> jnp.ndarray:
    total_sum_second_order_x2_x_kuu = jnp.zeros_like(deep_second_order_x_x_kuu(x1, x2, initial_theta, 0))
    num_x2 = x2.shape[1] - 1

    for d in range(num_x2):
        total_sum_second_order_x2_x_kuu += deep_second_order_x_x_kuu(x1, x2, initial_theta, d)

    first_order_term = deep_first_order_x_t_kuu(x1, x2, initial_theta, num_x2)
    kuu_term = react_diffs_high_d_kuu(x1, x2, initial_theta)
    nonlinear_term = kuu_term * kuu_term
    linear_term = kuu_term

    return first_order_term - total_sum_second_order_x2_x_kuu + nonlinear_term - linear_term


@jit
def react_diffs_high_d_kuf(x1: ArrayLike, x2: ArrayLike, initial_theta) -> jnp.ndarray:
    total_sum_second_order_x1_x_kuf = jnp.zeros_like(deep_second_order_x_x_kuu(x1, x2, initial_theta, 0))
    num_x1 = x1.shape[1] - 1

    for d in range(num_x1):
        total_sum_second_order_x1_x_kuf += deep_second_order_x_x_kuu(x1, x2, initial_theta, d)

    first_order_term = deep_first_order_x_t_kuu(x2, x1, initial_theta, num_x1)
    kuu_term = react_diffs_high_d_kuu(x1, x2, initial_theta)
    nonlinear_term = kuu_term * kuu_term
    linear_term = kuu_term

    return first_order_term - total_sum_second_order_x1_x_kuf + nonlinear_term - linear_term


@jit
def deep_first_order_x_t_kuu(x1: ArrayLike, x2: ArrayLike, initial_theta, idx: int) -> jnp.ndarray:
    lengthscale = initial_theta['lengthscale']
    sigma = initial_theta['sigma']
    diff = x1[:, idx] - x2[:, idx]
    return -sigma ** 2 / lengthscale ** 2 * diff * deep_rbf_kernel(x1, x2, initial_theta)


@jit
def deep_second_order_x_x_kuu(x1: ArrayLike, x2: ArrayLike, initial_theta, idx: int) -> jnp.ndarray:
    lengthscale = initial_theta['lengthscale']
    sigma = initial_theta['sigma']
    diff = x1[:, idx] - x2[:, idx]
    return (sigma ** 2 / lengthscale ** 4 * diff ** 2 - sigma ** 2 / lengthscale ** 2) * deep_rbf_kernel(x1, x2,
                                                                                                         initial_theta)


@jit
def deep_first_order_x_t_kuu(x1: ArrayLike, x2: ArrayLike, initial_theta, idx: int) -> jnp.ndarray:
    lengthscale = initial_theta['lengthscale']
    sigma = initial_theta['sigma']
    diff = x1[:, idx] - x2[:, idx]
    return -sigma ** 2 / lengthscale ** 2 * diff * deep_rbf_kernel(x1, x2, initial_theta)


@jit
def deep_second_order_x_x_kuu(x1: ArrayLike, x2: ArrayLike, initial_theta, idx: int) -> jnp.ndarray:
    lengthscale = initial_theta['lengthscale']
    sigma = initial_theta['sigma']
    diff = x1[:, idx] - x2[:, idx]
    return (sigma ** 2 / lengthscale ** 4 * diff ** 2 - sigma ** 2 / lengthscale ** 2) * deep_rbf_kernel(x1, x2,
                                                                                                         initial_theta)
