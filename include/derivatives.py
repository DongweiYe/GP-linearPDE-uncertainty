import jax.numpy as jnp
from jax import jvp, jit

from include.kernels import rbf_kernel, matern_kernel_3_2, matern_kernel_5_2, rational_quadratic_kernel, spectral_mixture_kernel



@jit
def first_order_x1_0(x1, x2, params):
    kernel = 'rbf'
    if kernel == 'rbf':
        def kernel_fn(x1_col):
            x1_mod = x1.at[:, 0].set(x1_col)
            return rbf_kernel(x1_mod, x2, params)
    elif kernel == 'matern_3_2':
        def kernel_fn(x1_col):
            x1_mod = x1.at[:, 0].set(x1_col)
            return matern_kernel_3_2(x1_mod, x2, params)
    elif kernel == 'matern_5_2':
        def kernel_fn(x1_col):
            x1_mod = x1.at[:, 0].set(x1_col)
            return matern_kernel_5_2(x1_mod, x2, params)
    elif kernel == 'rational_quadratic':
        def kernel_fn(x1_col):
            x1_mod = x1.at[:, 0].set(x1_col)
            return rational_quadratic_kernel(x1_mod, x2, params)
    elif kernel == 'spectral_mixture':
        def kernel_fn(x1_col):
            x1_mod = x1.at[:, 0].set(x1_col)
            return spectral_mixture_kernel(x1_mod, x2, params)
    else:
        raise ValueError('Invalid kernel specified')

    v = jnp.ones(len(x1[:, 0]))
    _, jvp_fn = jvp(kernel_fn, (x1[:, 0],), (v,))
    return jvp_fn


@jit
def first_order_x2_0(x1, x2, params):  # dx'
    kernel = 'rbf'
    if kernel == 'rbf':
        def kernel_fn(x2_col):
            x2_mod = x2.at[:, 0].set(x2_col)
            return rbf_kernel(x1, x2_mod, params)
    elif kernel == 'matern_3_2':
        def kernel_fn(x2_col):
            x2_mod = x2.at[:, 0].set(x2_col)
            return matern_kernel_3_2(x1, x2_mod, params)
    elif kernel == 'matern_5_2':
        def kernel_fn(x2_col):
            x2_mod = x2.at[:, 0].set(x2_col)
            return matern_kernel_5_2(x1, x2_mod, params)
    elif kernel == 'rational_quadratic':
        def kernel_fn(x2_col):
            x2_mod = x2.at[:, 0].set(x2_col)
            return rational_quadratic_kernel(x1, x2_mod, params)
    elif kernel == 'spectral_mixture':
        def kernel_fn(x2_col):
            x2_mod = x2.at[:, 0].set(x2_col)
            return spectral_mixture_kernel(x1, x2_mod, params)
    else:
        raise ValueError('Invalid kernel specified')

    v = jnp.ones(len(x2[:, 0]))
    _, jvp_fn = jvp(kernel_fn, (x2[:, 0],), (v,))
    return jvp_fn


@jit
def first_order_x1_1(x1, x2, params):  # dt
    kernel = 'rbf'
    if kernel == 'rbf':
        def kernel_fn(x1_col):
            x1_mod = x1.at[:, 1].set(x1_col)
            return rbf_kernel(x1_mod, x2, params)
    elif kernel == 'matern_3_2':
        def kernel_fn(x1_col):
            x1_mod = x1.at[:, 1].set(x1_col)
            return matern_kernel_3_2(x1_mod, x2, params)
    elif kernel == 'matern_5_2':
        def kernel_fn(x1_col):
            x1_mod = x1.at[:, 1].set(x1_col)
            return matern_kernel_5_2(x1_mod, x2, params)
    elif kernel == 'rational_quadratic':
        def kernel_fn(x1_col):
            x1_mod = x1.at[:, 1].set(x1_col)
            return rational_quadratic_kernel(x1_mod, x2, params)
    elif kernel == 'spectral_mixture':
        def kernel_fn(x1_col):
            x1_mod = x1.at[:, 1].set(x1_col)
            return spectral_mixture_kernel(x1_mod, x2, params)
    else:
        raise ValueError('Invalid kernel specified')

    v = jnp.ones(len(x1[:, 1]))
    _, jvp_fn = jvp(kernel_fn, (x1[:, 1],), (v,))
    return jvp_fn


@jit
def first_order_x2_1(x1, x2, params):  # dt'
    kernel = 'rbf'
    if kernel == 'rbf':
        def kernel_fn(x2_col):
            x2_mod = x2.at[:, 1].set(x2_col)
            return rbf_kernel(x1, x2_mod, params)
    elif kernel == 'matern_3_2':
        def kernel_fn(x2_col):
            x2_mod = x2.at[:, 1].set(x2_col)
            return matern_kernel_3_2(x1, x2_mod, params)
    elif kernel == 'matern_5_2':
        def kernel_fn(x2_col):
            x2_mod = x2.at[:, 1].set(x2_col)
            return matern_kernel_5_2(x1, x2_mod, params)
    elif kernel == 'rational_quadratic':
        def kernel_fn(x2_col):
            x2_mod = x2.at[:, 1].set(x2_col)
            return rational_quadratic_kernel(x1, x2_mod, params)
    elif kernel == 'spectral_mixture':
        def kernel_fn(x2_col):
            x2_mod = x2.at[:, 1].set(x2_col)
            return spectral_mixture_kernel(x1, x2_mod, params)
    else:
        raise ValueError('Invalid kernel specified')

    v = jnp.ones(len(x2[:, 1]))
    _, jvp_fn = jvp(kernel_fn, (x2[:, 1],), (v,))
    return jvp_fn


@jit
def second_order_x2_0_x2_0(x1, x2, params):  # dx' dx'
    v = jnp.ones(len(x2[:, 0]))

    def extract_fn(x2_col):
        x2_mod = x2.at[:, 0].set(x2_col)
        return first_order_x2_0(x1, x2_mod, params)

    _, jvp_fn = jvp(extract_fn, (x2[:, 0],), (v,))
    return jvp_fn


@jit
def second_order_x1_0_x1_0(x1, x2, params):  # dx dx
    v = jnp.ones(len(x1[:, 0]))

    def extract_fn(x1_col):
        x1_mod = x1.at[:, 0].set(x1_col)
        return first_order_x1_0(x1_mod, x2, params)

    _, jvp_fn = jvp(extract_fn, (x1[:, 0],), (v,))
    return jvp_fn


@jit
def second_order_x2_1_x1_1(x1, x2, params):  # dt'dt
    v = jnp.ones(len(x1[:, 1]))

    def extract_fn(x1_col):
        x1_mod = x1.at[:, 1].set(x1_col)
        return first_order_x2_1(x1_mod, x2, params)

    _, jvp_fn = jvp(extract_fn, (x1[:, 1],), (v,))
    return jvp_fn


@jit
def second_order_x2_1_x1_0(x1, x2, params):  # dt' dx
    v = jnp.ones(len(x1[:, 0]))

    def extract_fn(x1_col):
        x1_mod = x1.at[:, 0].set(x1_col)
        return first_order_x2_1(x1_mod, x2, params)

    _, jvp_fn = jvp(extract_fn, (x1[:, 0],), (v,))
    return jvp_fn


@jit
def third_order_x2_0_x2_0_x1_1(x1, x2, params):  # dx' dx' dt
    v = jnp.ones(len(x1[:, 1]))

    def extract_fn(x1_col):
        x1_mod = x1.at[:, 1].set(x1_col)
        return second_order_x2_0_x2_0(x1_mod, x2, params)

    _, jvp_fn = jvp(extract_fn, (x1[:, 1],), (v,))
    return jvp_fn


@jit
def third_order_x2_1_x1_0_x1_0(x1, x2, params):  # dt' dx dx
    v = jnp.ones(len(x1[:, 0]))

    def extract_fn(x1_col):
        x1_mod = x1.at[:, 0].set(x1_col)
        return second_order_x2_1_x1_0(x1_mod, x2, params)

    _, jvp_fn = jvp(extract_fn, (x1[:, 0],), (v,))
    return jvp_fn


@jit
def third_order_x2_0_x2_0_x1_0(x1, x2, params):  # dx' dx' dx
    v = jnp.ones(len(x1[:, 0]))

    def extract_fn(x1_col):
        x1_mod = x1.at[:, 0].set(x1_col)
        return second_order_x2_0_x2_0(x1_mod, x2, params)

    _, jvp_fn = jvp(extract_fn, (x1[:, 0],), (v,))
    return jvp_fn


@jit
def fourth_order_x2_0_x2_0_x1_0_x1_0(x1, x2, params):  # dx' dx' dx dx
    v = jnp.ones(len(x1[:, 0]))

    def extract_fn(x1_col):
        x1_mod = x1.at[:, 0].set(x1_col)
        return third_order_x2_0_x2_0_x1_0(x1_mod, x2, params)

    _, jvp_fn = jvp(extract_fn, (x1[:, 0],), (v,))
    return jvp_fn


@jit
def first_order_x1(x1, x2, params, init_d):
    d = init_d

    def kernel_fn(x1_col):
        x1_mod = x1.at[:, d].set(x1_col)
        return rbf_kernel(x1_mod, x2, params)

    v = jnp.ones(len(x1[:, d]))
    _, jvp_fn = jvp(kernel_fn, (x1[:, d],), (v,))
    return jvp_fn


@jit
def first_order_x2(x1, x2, params, init_d):
    d = init_d

    def kernel_fn(x2_col):
        x2_mod = x2.at[:, d].set(x2_col)
        return rbf_kernel(x1, x2_mod, params)

    v = jnp.ones(len(x2[:, d]))
    _, jvp_fn = jvp(kernel_fn, (x2[:, d],), (v,))
    return jvp_fn

@jit
def first_order_x2_t_kuu(x1, x2, params):
    return first_order_x2(x1, x2, params, -1)


@jit
def first_order_x1_t_kuu(x1, x2, params):
    return first_order_x1(x1, x2, params, -1)


@jit
def second_order_x1(x1, x2, params, init_d):
    d = init_d
    v = jnp.ones(len(x1[:, d]))

    def extract_fn(x1_col):
        x1_mod = x1.at[:, d].set(x1_col)
        return first_order_x1(x1_mod, x2, params, d)

    _, jvp_fn = jvp(extract_fn, (x1[:, d],), (v,))
    return jvp_fn


@jit
def second_order_x2(x1, x2, params, init_d):
    d = init_d
    v = jnp.ones(len(x2[:, d]))

    def extract_fn(x2_col):
        x2_mod = x2.at[:, d].set(x2_col)
        return first_order_x2(x1, x2_mod, params, d)

    _, jvp_fn = jvp(extract_fn, (x2[:, d],), (v,))
    return jvp_fn


