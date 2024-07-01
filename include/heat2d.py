import datetime
from scipy.stats import qmc
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit
from math import pi
from include.derivatives import first_order_x1_1, first_order_x2_1, second_order_x1_0_x1_0, second_order_x2_0_x2_0, \
    second_order_x2_1_x1_1, third_order_x2_0_x2_0_x1_1, third_order_x2_1_x1_0_x1_0, fourth_order_x2_0_x2_0_x1_0_x1_0
from include.kernels import rbf_kernel
from scipy.stats import truncnorm

plt.rcParams.update({"figure.figsize": (12, 6)})
plt.rcParams.update({'font.size': 22})

def f_xt(Xf) -> jnp.ndarray:
    x = Xf[:, :1]
    t = Xf[:, -1:]
    f: jnp.ndarray = jnp.exp((-1) * t) * (4 * (pi ** 2) + (-1)) * jnp.sin(2 * pi * x)
    # f: jnp.ndarray = jnp.exp((-1) * t) * ((1/4) * (pi ** 2) + (-1)) * jnp.sin((pi/2) * x)
    return f


def u_xt(Xu_fixed) -> jnp.ndarray:
    x = Xu_fixed[:, :1]
    t = Xu_fixed[:, -1:]
    u: jnp.ndarray = jnp.exp((-1) * t) * jnp.sin(2 * pi * x)
    # u: jnp.ndarray = jnp.exp((-1) * t) * jnp.sin((pi/2) * x)
    return u

def plot_u_f(Xu_certain_all, Xf, Xu_noise):
    x = jnp.linspace(0, 1, 100)
    t = jnp.linspace(0, 1, 100)
    X, T = jnp.meshgrid(x, t)
    X_plot = jnp.vstack([X.ravel(), T.ravel()]).T

    f_values = f_xt(X_plot).reshape(X.shape)
    u_values = u_xt(X_plot).reshape(X.shape)

    # Plot the functions with enhanced aesthetics
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # f(x, t) plot
    c1 = ax[0].contourf(X, T, f_values, levels=50, cmap='viridis')
    fig.colorbar(c1, ax=ax[0], orientation='vertical', label='f(x, t) value')
    ax[0].scatter(Xf[:, 0], Xf[:, 1], color='black', label='Xu points')
    ax[0].set_title('f(x, t)', fontsize=16)
    ax[0].set_xlabel('x', fontsize=14)
    ax[0].set_ylabel('t', fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=12)

    # u(x, t) plot
    c2 = ax[1].contourf(X, T, u_values, levels=50, cmap='plasma')
    fig.colorbar(c2, ax=ax[1], orientation='vertical', label='u(x, t) value')
    ax[1].scatter(Xu_certain_all[:, 0], Xu_certain_all[:, 1], color='black', label='Xu points')
    ax[1].scatter(Xu_noise[:, 0], Xu_noise[:, 1], color='red', label='Xu noise points', marker='x')
    ax[1].set_title('u(x, t)', fontsize=16)
    ax[1].set_xlabel('x', fontsize=14)
    ax[1].set_ylabel('t', fontsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=12)

    # General layout adjustments
    for a in ax:
        a.set_aspect('auto')
        a.grid(True, linestyle='--', alpha=0.6)
    current_time = datetime.datetime.now().strftime("%M%S")
    plt.suptitle('Function Plots of f(x, t) and u(x, t)', fontsize=18)
    plt.savefig(f'u_f_plot_{current_time}.png')
# def u_xt_noise(Xu_noise) -> jnp.ndarray:
#     x = Xu_noise[:, :1]
#     t = Xu_noise[:, -1:]
#     u: jnp.ndarray = jnp.exp((-1) * t) * jnp.sin(2 * pi * x)
#     noise_std = 4e-2
#     u_noise = u + noise_std * jax.random.normal(jax.random.PRNGKey(0), shape=u.shape)
#     return u_noise


def get_u_training_data_2d(key_x_u, key_x_u_init, key_t_u_low, key_t_u_high, key_x_noise, key_t_noise, sample_num,
                           init_num, bnum, noise_std) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                              jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray):
    # noise_std = 1e-1
    Xu_certain = jax.random.uniform(key_x_u, shape=(sample_num, 2), dtype=jnp.float64)
    yu_certain = u_xt(Xu_certain)
    # yu_noise = u_xt_noise(Xu_certain)
    xu, tu = Xu_certain[:, :1], Xu_certain[:, -1:]

    xu_noise = xu + noise_std * jax.random.normal(key_x_noise, shape=xu.shape)
    tu_noise = tu + noise_std * jax.random.normal(key_t_noise, shape=tu.shape)
    # xu_noise = jnp.clip(xu_noise, 0, 1)
    # tu_noise = jnp.clip(tu_noise, 0, 1)

    Xu_noise = jnp.concatenate([xu_noise, tu_noise], axis=1)
    # init + boundary
    xu_init = jax.random.uniform(key_x_u_init, shape=(init_num, 1), dtype=jnp.float64)
    tu_init = jnp.zeros(shape=(init_num, 1))
    # Xu_init = jnp.concatenate([xu_init, tu_init], axis=1)

    xu_bound_low = jnp.zeros(shape=(bnum, 1))
    xu_bound_high = jnp.ones(shape=(bnum, 1))
    tu_bound_low = jax.random.uniform(key_t_u_low, shape=(bnum, 1), dtype=jnp.float64)
    tu_bound_high = jax.random.uniform(key_t_u_high, shape=(bnum, 1), dtype=jnp.float64)

    xu_fixed = jnp.concatenate((xu_bound_low, xu_init, xu_bound_high))
    tu_fixed = jnp.concatenate((tu_bound_low, tu_init, tu_bound_high))
    Xu_fixed = jnp.concatenate([xu_fixed, tu_fixed], axis=1)
    print("Xu_fixed: ", Xu_fixed)
    Yu_fixed = u_xt(Xu_fixed)

    return Xu_certain, yu_certain, xu_noise, tu_noise, Xu_noise, xu_fixed, tu_fixed, Xu_fixed, Yu_fixed


def get_u_training_data_2d_qmc(key_x_u, key_x_u_init, key_t_u_low, key_t_u_high, key_x_noise, key_t_noise, sample_num,
                           init_num, bnum, noise_std) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                              jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray):
    qMCsampler = qmc.Sobol(d=2, seed=0)
    qMCsample = qMCsampler.random_base2(m=int(jnp.log2(sample_num)))  # 生成最近的2的幂次个样本
    if qMCsample.shape[0] > sample_num:
        qMCsample = qMCsample[:sample_num]

    Xu_certain = jnp.array(qMCsample)
    yu_certain = u_xt(Xu_certain)
    # yu_noise = u_xt_noise(Xu_certain)
    xu, tu = Xu_certain[:, :1], Xu_certain[:, -1:]

    xu_noise = xu + noise_std * jax.random.normal(key_x_noise, shape=xu.shape)
    tu_noise = tu + noise_std * jax.random.normal(key_t_noise, shape=tu.shape)
    # xu_noise = jnp.clip(xu_noise, 0, 1)
    # tu_noise = jnp.clip(tu_noise, 0, 1)
    Xu_noise = jnp.concatenate([xu_noise, tu_noise], axis=1)


    # init + boundary
    qmc_sampler_init = qmc.Sobol(d=1, seed=1)
    qmc_sampler_bound_low = qmc.Sobol(d=1, seed=2)
    qmc_sampler_bound_high = qmc.Sobol(d=1, seed=3)
    xu_init = jnp.array(qmc_sampler_init.random_base2(m=int(jnp.log2(init_num))))
    init_num = xu_init.shape[0]
    tu_init = jnp.zeros(shape=(init_num, 1))

    tu_bound_low = jnp.array(qmc_sampler_bound_low.random_base2(m=int(jnp.log2(bnum))))
    tu_bound_high = jnp.array(qmc_sampler_bound_high.random_base2(m=int(jnp.log2(bnum))))
    bnum = tu_bound_low.shape[0]
    xu_bound_low = jnp.zeros(shape=(bnum, 1))
    xu_bound_high = jnp.ones(shape=(bnum, 1))

    xu_fixed = jnp.concatenate((xu_bound_low, xu_init, xu_bound_high))
    tu_fixed = jnp.concatenate((tu_bound_low, tu_init, tu_bound_high))
    Xu_fixed = jnp.concatenate([xu_fixed, tu_fixed], axis=1)
    print("Xu_fixed: ", Xu_fixed)
    Yu_fixed = u_xt(Xu_fixed)

    return Xu_certain, yu_certain, xu_noise, tu_noise, Xu_noise, xu_fixed, tu_fixed, Xu_fixed, Yu_fixed, init_num, bnum


def get_f_training_data_2d(key_x_f, sample_num) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray):
    # Xf = jax.random.uniform(key_x_f, shape=(sample_num, 2), dtype=jnp.float64)
    qMCsampler = qmc.Sobol(d=2, seed=4)
    qMCsample = qMCsampler.random_base2(m=int(jnp.log2(sample_num)))
    if qMCsample.shape[0] > sample_num:
        qMCsample = qMCsample[:sample_num]
    Xf = jnp.array(qMCsample)
    yf = f_xt(Xf)
    xf, tf = Xf[:, :1], Xf[:, -1:]

    return xf, tf, Xf, yf

@jit
def heat_equation_kuu(x1, x2, params) -> jnp.ndarray:
    return rbf_kernel(x1, x2, params)


@jit
def heat_equation_kuu_noise(x1, x2, params) -> jnp.ndarray:
    noise_variance = 1e-2
    kzz = rbf_kernel(x1, x2, params)
    noise_term = noise_variance * jnp.eye(x1.shape[0])
    return kzz + noise_term


@jit
def heat_equation_kuu_noise2(x1, x2, params) -> jnp.ndarray:
    noise_variance = 1e-4
    kzz = rbf_kernel(x1, x2, params)
    noise_term = noise_variance * jnp.eye(x1.shape[0])
    return kzz + noise_term


@jit
def heat_equation_kuf(x1, x2, params) -> jnp.ndarray:
    return first_order_x2_1(x1, x2, params) - second_order_x2_0_x2_0(x1, x2, params)

@jit
def heat_equation_kfu(x1, x2, params) -> jnp.ndarray:
    return first_order_x1_1(x1, x2, params) - second_order_x1_0_x1_0(x1, x2, params)

@jit
def heat_equation_kff(x1, x2, params) -> jnp.ndarray:
    _1 = second_order_x2_1_x1_1(x1, x2, params)
    _2 = third_order_x2_0_x2_0_x1_1(x1, x2, params)
    _3 = third_order_x2_1_x1_0_x1_0(x1, x2, params)
    _4 = fourth_order_x2_0_x2_0_x1_0_x1_0(x1, x2, params)
    return _1 - _2 - _3 + _4

@jit
def heat_equation_kff_noise(x1, x2, params) -> jnp.ndarray:
    noise_variance = 1e-4
    kff = heat_equation_kff(x1, x2, params)
    noise_term = noise_variance * jnp.eye(x1.shape[0])
    return kff + noise_term



def heat_equation_nlml_loss_2d(heat_params, Xuz, Xfz, Xfg, number_Y, Y) -> float:
    number_Y = number_Y
    init = heat_params
    Xuz, Xfz, Xfg = Xuz, Xfz, Xfg
    params = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    zz_uu = heat_equation_kuu(Xuz, Xuz, params)
    zz_uf = heat_equation_kuu(Xuz, Xfz, params)
    zg_uf = heat_equation_kuf(Xuz, Xfg, params)
    zz_fu = heat_equation_kuu(Xfz, Xuz, params)
    zz_ff = heat_equation_kuu(Xfz, Xfz, params)
    zg_ff = heat_equation_kuf(Xfz, Xfg, params)
    gz_fu = heat_equation_kfu(Xfg, Xuz, params)
    gz_ff = heat_equation_kfu(Xfg, Xfz, params)
    gg_ff = heat_equation_kff(Xfg, Xfg, params)
    # print("zz_uu: ", zz_uu)
    # print("zz_uf: ", zz_uf)
    # print("zg_uf: ", zg_uf)
    # print("zz_fu: ", zz_fu)
    # print("zz_ff: ", zz_ff)
    # print("zg_ff: ", zg_ff)
    # print("gz_fu: ", gz_fu)
    # print("gz_ff: ", gz_ff)
    # print("gg_ff: ", gg_ff)

    K = jnp.block([[zz_uu, zz_uf, zg_uf], [zz_fu, zz_ff, zg_ff], [gz_fu, gz_ff, gg_ff]])
    sign, logdet = jnp.linalg.slogdet(K)
    K_inv_Y = jnp.linalg.solve(K, Y)
    signed_logdet = sign * logdet
    K_inv_Y_product = Y.T @ K_inv_Y
    scalar_result = jnp.squeeze(K_inv_Y_product)
    nlml = (1 / 2 * signed_logdet) + (1 / 2 * scalar_result) + ((number_Y / 2) * jnp.log(2 * jnp.pi))

    # print("K: ", K)
    # print("K.shape: ", K.shape)
    # print("logdet: ", logdet)
    # print("K_inv_Y: ", K_inv_Y)
    # print("signed_logdet: ", signed_logdet)
    # print("Y.T @ K_inv_Y: ", Y.T @ K_inv_Y)
    # print("Y.T @ K_inv_Y shape: ", (Y.T @ K_inv_Y).shape)
    # print("scalar_result: ", scalar_result)
    # print("scalar_result.shape: ", scalar_result.shape)
    # print("nlml: ", nlml)
    # print("nlml.shape: ", nlml.shape)
    return  nlml


