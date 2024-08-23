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
from jax.typing import ArrayLike
from jax import jit, jvp
from typing import Any, Dict

plt.rcParams.update({"figure.figsize": (12, 6)})
plt.rcParams.update({'font.size': 22})

################################################# functions ##########################################################
def f_xt(Xf) -> jnp.ndarray:
    x = Xf[:, :1]
    t = Xf[:, -1:]
    # f: jnp.ndarray = jnp.exp((-1) * t) * (4 * (pi ** 2) + (-1)) * jnp.sin(2 * pi * x)
    # f: jnp.ndarray = jnp.exp((-1) * t) * ((1/4) * (pi ** 2) + (-1)) * jnp.sin((pi/2) * x)
    f: jnp.ndarray = jnp.exp((-1) * t) * (16 * (pi ** 2) + (-1)) * jnp.sin(4 * pi * x)
    return f


def u_xt(Xu_fixed) -> jnp.ndarray:
    x = Xu_fixed[:, :1]
    t = Xu_fixed[:, -1:]
    # u: jnp.ndarray = jnp.exp((-1) * t) * jnp.sin(2 * pi * x)
    # u: jnp.ndarray = jnp.exp((-1) * t) * jnp.sin((pi/2) * x)
    u: jnp.ndarray = jnp.exp((-1) * t) * jnp.sin(4 * pi * x)
    return u

################################################# kernels ############################################################
@jit
def deep_rbf_kernel(x1: ArrayLike, x2: ArrayLike, initial_theta) -> jnp.ndarray:
    lengthscale = initial_theta['lengthscale']
    sigma = initial_theta['sigma']
    square_distances = jnp.sum((x1[:, jnp.newaxis, :] - x2[jnp.newaxis, :, :]) ** 2 / lengthscale ** 2,
                               axis=-1)
    rbf = sigma ** 2 * jnp.exp(-0.5 * square_distances)
    return rbf

@jit
def compute_kuu(x1: ArrayLike, x2: ArrayLike, params) -> jnp.ndarray:
    return deep_rbf_kernel(x1, x2, params)


@jit
def compute_kuf(x1: jnp.ndarray, x2: jnp.ndarray, initial_theta, lengthscale_x, lengthscale_t) -> jnp.ndarray:
    # lengthscale_x = initial_theta[0][1][0].item()
    # lengthscale_t = initial_theta[0][1][1].item()

    t1 = x1[:, -1]
    t2 = x2[:, -1]
    x1_spatial = x1[:, :-1]
    x2_spatial = x2[:, :-1]

    params = {'sigma': initial_theta[-1][0], 'lengthscale': initial_theta[-1][1]}
    k = deep_rbf_kernel(x1, x2, params)

    term1 = (t1[:, None] - t2[None, :]) / lengthscale_t ** 2
    term2 = jnp.sum((x1_spatial[:, None] - x2_spatial[None, :]) ** 2, axis=-1) / lengthscale_x ** 4
    term3 = 1 / lengthscale_x ** 2

    k_uf = k * (term1 + term3 - term2)

    return k_uf

@jit
def compute_kfu(x1: jnp.ndarray, x2: jnp.ndarray, initial_theta, lengthscale_x, lengthscale_t) -> jnp.ndarray:
    # lengthscale_x = initial_theta[0][1][0].item()
    # lengthscale_t = initial_theta[0][1][1].item()
    t1 = x1[:, -1]
    t2 = x2[:, -1]
    x1_spatial = x1[:, :-1]
    x2_spatial = x2[:, :-1]

    params = {'sigma': initial_theta[-1][0], 'lengthscale': initial_theta[-1][1]}
    k = deep_rbf_kernel(x1, x2, params)

    term1 = -(t1[:, None] - t2[None, :]) / lengthscale_t ** 2
    term2 = jnp.sum((x2_spatial[None] - x1_spatial[:, None]) ** 2, axis=-1) / lengthscale_x ** 4
    term3 = 1 / lengthscale_x ** 2

    k_fu = k * (term1 - term2 + term3)

    return k_fu


@jit
def compute_kff(x1: jnp.ndarray, x2: jnp.ndarray, initial_theta, lengthscale_x, lengthscale_t) -> jnp.ndarray:
    # lengthscale_x = initial_theta[0][1][0].item()
    # lengthscale_t = initial_theta[0][1][1].item()
    t1 = x1[:, -1]
    t2 = x2[:, -1]
    x1_spatial = x1[:, :-1]
    x2_spatial = x2[:, :-1]

    params = {'sigma': initial_theta[-1][0], 'lengthscale': initial_theta[-1][1]}
    k = deep_rbf_kernel(x1, x2, params)

    term1 = k * ( (t1[:, None] - t2) / lengthscale_t ** 2 ) * (
            -(t1[:, None] - t2) / lengthscale_t ** 2
            - jnp.sum((x2_spatial[None] - x1_spatial[:, None]) ** 2, axis=-1) / lengthscale_x ** 4
            + 1 / lengthscale_x ** 2
        ) + k / lengthscale_t ** 2

    term2 = (2 * k / lengthscale_x ** 6) * jnp.sum((x1_spatial[:, None] - x2_spatial[None]) ** 2, axis=-1) - (2 * k / lengthscale_x ** 4)

    term3 = -k * (t2 - t1[:, None]) / (lengthscale_x ** 2 * lengthscale_t ** 2) + k * jnp.sum((x1_spatial[:, None] - x2_spatial[None]) ** 2, axis=-1) * (t2 - t1[:, None]) / (lengthscale_x ** 4 * lengthscale_t ** 2)

    term4 = -k / lengthscale_x ** 4 + k * jnp.sum((x1_spatial[:, None] - x2_spatial[None]) ** 2, axis=-1) / lengthscale_x ** 6

    term5 = k * (3 * jnp.sum((x2_spatial[None] - x1_spatial[:, None]) ** 2, axis=-1) / lengthscale_x ** 6 - jnp.sum((x2_spatial[None] - x1_spatial[:, None]) ** 4, axis=-1) / lengthscale_x ** 8)

    total_expression = term1 - ( term2 + term3 + term4 + term5)

    return total_expression
################################################# nlml ##########################################################
def heat_equation_nlml_loss_2d(heat_params, Xuz, Xfz, Xfg, number_Y, Y) -> float:
    init = heat_params
    params =  heat_params
    params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    lengthscale_x = params[0][1][0].item()
    lengthscale_t = params[0][1][1].item()

    zz_uu = compute_kuu(Xuz, Xuz, params_kuu)
    zz_uf = compute_kuu(Xuz, Xfz, params_kuu)
    zg_uf = compute_kuf(Xuz, Xfg, params, lengthscale_x, lengthscale_t)
    zz_fu = compute_kuu(Xfz, Xuz, params_kuu)
    zz_ff = compute_kuu(Xfz, Xfz, params_kuu)
    zg_ff = compute_kuf(Xfz, Xfg, params, lengthscale_x, lengthscale_t)
    gz_fu = compute_kfu(Xfg, Xuz, params, lengthscale_x, lengthscale_t)
    gz_ff = compute_kfu(Xfg, Xfz, params, lengthscale_x, lengthscale_t)
    gg_ff = compute_kff(Xfg, Xfg, params, lengthscale_x, lengthscale_t)
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

################################################ plot ############################################################
def plot_u_f(Xu_certain_all, Xf, Xu_noise, noise_std):
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
    ax[1].scatter(Xu_noise[:, 0], Xu_noise[:, 1], color='blue', label='Xu noise points', marker='o')
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
    plt.savefig(f'u_f_plot_{noise_std}_{current_time}.png')
# def u_xt_noise(Xu_noise) -> jnp.ndarray:
#     x = Xu_noise[:, :1]
#     t = Xu_noise[:, -1:]
#     u: jnp.ndarray = jnp.exp((-1) * t) * jnp.sin(2 * pi * x)
#     noise_std = 4e-2
#     u_noise = u + noise_std * jax.random.normal(jax.random.PRNGKey(0), shape=u.shape)
#     return u_noise

def plot_u_f_pred(Xu_certain_all, Xf, Xu_noise, noise_std, Xu_pred, prior_var,assumption_sigma,k,max_samples):
    x = jnp.linspace(0, 1, 100)
    t = jnp.linspace(0, 1, 100)
    X, T = jnp.meshgrid(x, t)
    X_plot = jnp.vstack([X.ravel(), T.ravel()]).T

    f_values = f_xt(X_plot).reshape(X.shape)
    u_values = u_xt(X_plot).reshape(X.shape)

    num_f = Xf.shape[0]

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
    ax[1].scatter(Xu_certain_all[:, 0], Xu_certain_all[:, 1], color='black', label='GT points')
    ax[1].scatter(Xu_noise[:, 0], Xu_noise[:, 1], color='tab:blue', label='Xu noise', marker='x')
    ax[1].scatter(Xu_pred[:, 0], Xu_pred[:, 1], color='tab:red', label='Xu infer', marker='x')
    ax[1].set_title('u(x, t)', fontsize=16)
    ax[1].set_xlabel('x', fontsize=14)
    ax[1].set_ylabel('t', fontsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    for a in ax:
        a.set_aspect('auto')
        a.grid(True, linestyle='--', alpha=0.6)
    current_time = datetime.datetime.now().strftime("%M%S")
    # plt.suptitle('Function Plots of f(x, t) and u(x, t)', fontsize=18)
    plt.savefig(f'pred_k{k}_priorvar_{prior_var}_assump{assumption_sigma}_nstd{noise_std}_iter{max_samples}_{current_time}.png')


def plot_u_pred(Xu_certain_all, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred, prior_var,assumption_sigma,k,max_samples,learning,num_chains,number_f,added_text):
    x = jnp.linspace(0, 1, 100)
    t = jnp.linspace(0, 1, 100)
    X, T = jnp.meshgrid(x, t)
    X_plot = jnp.vstack([X.ravel(), T.ravel()]).T
    u_values = u_xt(X_plot).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # u(x, t) plot
    c1 = ax.contourf(X, T, u_values, levels=50, cmap='plasma')
    # c1 = ax.contour(X, T, u_values, levels=50, cmap='plasma', alpha=0.5)  # Using contour for line contours
    # ax.contourf(X, T, u_values, levels=50, cmap='plasma', alpha=0.3)  # Using contourf for filled contours
    fig.colorbar(c1, ax=ax, orientation='vertical', label='u(x, t) value')
    #ax.scatter(Xu_certain_all[:, 0], Xu_certain_all[:, 1], color='black', label='GT', marker='o')
    ax.scatter(Xu_certain[:, 0], Xu_certain[:, 1], color='black', label='GT', marker='o')
    ax.scatter(Xu_noise[:, 0], Xu_noise[:, 1], color='tab:blue', label='Xu noise', marker='x')
    ax.scatter(Xu_pred[:, 0], Xu_pred[:, 1], color='tab:red', label='Posterior', marker='o')
    ax.set_title('u(x, t)', fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('t', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.legend(loc='upper right', fontsize=12)
    ax.set_aspect('auto')
    ax.grid(True, linestyle='--', alpha=0.6)

    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f'contourf_{added_text}.png')

def plot_u_pred_rd(Xu_certain_all, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred, prior_var,assumption_sigma,k,max_samples,learning,num_chains,number_f,added_text):
    x = jnp.linspace(-1, 1, 100)
    t = jnp.linspace(0, 1, 100)
    X, T = jnp.meshgrid(x, t)
    X_plot = jnp.vstack([X.ravel(), T.ravel()]).T
    u_values = u_xt(X_plot).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # u(x, t) plot
    c1 = ax.contourf(X, T, u_values, levels=50, cmap='plasma')
    # c1 = ax.contour(X, T, u_values, levels=50, cmap='plasma', alpha=0.5)  # Using contour for line contours
    # ax.contourf(X, T, u_values, levels=50, cmap='plasma', alpha=0.3)  # Using contourf for filled contours
    fig.colorbar(c1, ax=ax, orientation='vertical', label='u(x, t) value')
    #ax.scatter(Xu_certain_all[:, 0], Xu_certain_all[:, 1], color='black', label='GT', marker='o')
    ax.scatter(Xu_certain[:, 0], Xu_certain[:, 1], color='black', label='GT', marker='o')
    ax.scatter(Xu_noise[:, 0], Xu_noise[:, 1], color='tab:blue', label='Xu noise', marker='x')
    ax.scatter(Xu_pred[:, 0], Xu_pred[:, 1], color='tab:red', label='Posterior', marker='o')
    ax.set_title('u(x, t)', fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('t', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.legend(loc='upper right', fontsize=12)
    ax.set_aspect('auto')
    ax.grid(True, linestyle='--', alpha=0.6)

    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f'contourf_{added_text}.png')

def get_u_training_data_2d(key_x_u, key_x_u_init, key_t_u_low, key_t_u_high, key_x_noise, key_t_noise, sample_num,
                           init_num, bnum, noise_std) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                              jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray):
    # noise_std = 1e-1
    Xu_certain = jax.random.uniform(key_x_u, shape=(sample_num, 2), dtype=jnp.float64)
    yu_certain = u_xt(Xu_certain)
    # yu_noise = u_xt_noise(Xu_certain)
    xu, tu = Xu_certain[:, :1], Xu_certain[:, -1:]

    xu_noise = xu -jnp.abs(noise_std * jax.random.normal(key_x_noise, shape=xu.shape))
    tu_noise = tu -jnp.abs(noise_std * jax.random.normal(key_t_noise, shape=tu.shape))
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
                           init_num, bnum, noise_std, number_u_only_x) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                              jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray):
    qMCsampler = qmc.Sobol(d=2, seed=0)
    sample_num_total = sample_num + number_u_only_x
    qMCsample = qMCsampler.random_base2(m=int(jnp.round(jnp.log2(sample_num_total))))
    if qMCsample.shape[0] > sample_num_total:
        qMCsample = qMCsample[:sample_num_total]

    Xu_certain = jnp.array(qMCsample)
    print("qMCsample: ", qMCsample)
    print("number of xu_certain: ", Xu_certain.shape[0])
    yu_certain = u_xt(Xu_certain)
    # yu_noise = u_xt_noise(Xu_certain)
    xu, tu = Xu_certain[:, :1], Xu_certain[:, -1:]

    xu_noise = xu + noise_std * jax.random.normal(key_x_noise, shape=xu.shape)
    tu_with_noise = tu[:sample_num] + noise_std * jax.random.normal(key_t_noise, shape=tu[:sample_num].shape)
    tu_noise = jnp.concatenate([tu_with_noise, tu[sample_num:]])
    Xu_noise = jnp.concatenate([xu_noise, tu_noise], axis=1)
    Xu_noise = jnp.maximum(0, jnp.minimum(1, Xu_noise))
    #Xu_noise = jnp.clip(Xu_noise, 0, 1)
    print("Xu_noise: ", Xu_noise)
    print("tu: ", tu)
    print("tu_noise: ", tu_noise)
    print("xu_noise: ", xu_noise)
    print("xu: ", xu)

    # init + boundary
    # qmc_sampler_init = qmc.Sobol(d=1, seed=1)
    # qmc_sampler_bound_low = qmc.Sobol(d=1, seed=2)
    # qmc_sampler_bound_high = qmc.Sobol(d=1, seed=3)
    # xu_init = jnp.array(qmc_sampler_init.random_base2(m=int(jnp.log2(init_num))))
    # init_num = xu_init.shape[0]
    # tu_init = jnp.zeros(shape=(init_num, 1))
    #
    # tu_bound_low = jnp.array(qmc_sampler_bound_low.random_base2(m=int(jnp.log2(bnum))))
    # tu_bound_high = jnp.array(qmc_sampler_bound_high.random_base2(m=int(jnp.log2(bnum))))
    # bnum = tu_bound_low.shape[0]
    # xu_bound_low = jnp.zeros(shape=(bnum, 1))
    # xu_bound_high = jnp.ones(shape=(bnum, 1))
    #
    # xu_fixed = jnp.concatenate((xu_bound_low, xu_init, xu_bound_high))
    # tu_fixed = jnp.concatenate((tu_bound_low, tu_init, tu_bound_high))
    # Xu_fixed = jnp.concatenate([xu_fixed, tu_fixed], axis=1)
    xu_init = ((jnp.cos(jnp.arange(init_num + 1) * jnp.pi / init_num)) + 1) / 2
    xu_init = xu_init[1:-1, ]
    xu_init = jnp.expand_dims(xu_init, axis=-1)
    print("xu_init.shape", xu_init.shape)
    init_num = xu_init.shape[0]
    tu_init = jnp.zeros(shape=(init_num, 1))
    print("tu_init.shape", tu_init.shape)

    tu_bound_low = ((jnp.cos(jnp.arange(bnum + 1) * jnp.pi / bnum)) + 1) / 2
    tu_bound_low = jnp.expand_dims(tu_bound_low, axis=-1)
    print("tu_bound_low.shape", tu_bound_low.shape)
    tu_bound_high = ((jnp.cos(jnp.arange(bnum + 1) * jnp.pi / bnum)) + 1) / 2
    tu_bound_high = jnp.expand_dims(tu_bound_high, axis=-1)
    print("tu_bound_high.shape", tu_bound_high.shape)
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
    # qMCsampler = qmc.Sobol(d=2, seed=4)
    # qMCsample = qMCsampler.random_base2(m=int(jnp.log2(sample_num)))
    # if qMCsample.shape[0] > sample_num:
    #     qMCsample = qMCsample[:sample_num]
    # Xf = jnp.array(qMCsample)
    #
    # xf, tf = Xf[:, :1], Xf[:, -1:]

    xf =((jnp.cos(jnp.arange(sample_num+1) * jnp.pi /sample_num))+1 )/ 2
    tf = xf
    Xf, Tf = jnp.meshgrid(xf, tf)
    X_f = jnp.vstack([Xf.ravel(), Tf.ravel()]).T
    yf = f_xt(X_f)
    return xf, tf, X_f, yf

def get_u_test_data_2d_qmc(sample_num):
    qMCsampler = qmc.Sobol(d=2, seed=5)
    qMCsample = qMCsampler.random_base2(m=int(jnp.log2(sample_num)))
    if qMCsample.shape[0] > sample_num:
        qMCsample = qMCsample[:sample_num]
    X_test = jnp.array(qMCsample)
    y_test = u_xt(X_test)
    return X_test, y_test


################################################# old deep kernels ##########################################################
# @jit
# def heat_equation_kuu_noise(x1, x2, params) -> jnp.ndarray:
#     noise_variance = 1e-2
#     kzz = rbf_kernel(x1, x2, params)
#     noise_term = noise_variance * jnp.eye(x1.shape[0])
#     return kzz + noise_term

#
# @jit
# def heat_equation_kuu_noise2(x1, x2, params) -> jnp.ndarray:
#     noise_variance = 1e-4
#     kzz = rbf_kernel(x1, x2, params)
#     noise_term = noise_variance * jnp.eye(x1.shape[0])
#     return kzz + noise_term

# @jit
# def heat_equation_kuu(x1, x2, params) -> jnp.ndarray:
#     return rbf_kernel(x1, x2, params)
# @jit
# def heat_equation_kuf(x1, x2, params) -> jnp.ndarray:
#     return first_order_x2_1(x1, x2, params) - second_order_x2_0_x2_0(x1, x2, params)
#
# @jit
# def heat_equation_kfu(x1, x2, params) -> jnp.ndarray:
#     return first_order_x1_1(x1, x2, params) - second_order_x1_0_x1_0(x1, x2, params)
#
# @jit
# def heat_equation_kff(x1, x2, params) -> jnp.ndarray:
#     _1 = second_order_x2_1_x1_1(x1, x2, params)
#     _2 = third_order_x2_0_x2_0_x1_1(x1, x2, params)
#     _3 = third_order_x2_1_x1_0_x1_0(x1, x2, params)
#     _4 = fourth_order_x2_0_x2_0_x1_0_x1_0(x1, x2, params)
#     return _1 - _2 - _3 + _4

# @jit
# def heat_equation_kff_noise(x1, x2, params) -> jnp.ndarray:
#     noise_variance = 1e-4
#     kff = heat_equation_kff(x1, x2, params)
#     noise_term = noise_variance * jnp.eye(x1.shape[0])
#     return kff + noise_term
#




#
# @jit
# def heat_equation_kff(x1: ArrayLike, x2: ArrayLike, params) -> jnp.ndarray:
#     total_sum_second_order_x1_x_kuf = jnp.zeros_like(deep_second_order_x1_x_kuf(x1, x2, params, 0))
#     num_x = x1.shape[1] - 1
#     for d in range(num_x):
#         total_sum_second_order_x1_x_kuf += deep_second_order_x1_x_kuf(x1, x2, params, d)
#     return deep_first_order_x1_t_kuf(x1, x2, params) - total_sum_second_order_x1_x_kuf
#
#
#
# @jit
# def heat_equation_kuf(x1: ArrayLike, x2: ArrayLike, params) -> jnp.ndarray:
#     total_sum_second_order_x2_x_kuu = jnp.zeros_like(deep_second_order_x2_x_kuu(x1, x2, params, 0))
#     num_x2 = x2.shape[1] - 1
#     for d in range(num_x2):
#         total_sum_second_order_x2_x_kuu += deep_second_order_x2_x_kuu(x1, x2, params, d)
#     return deep_first_order_x2_t_kuu(x1, x2, params) - total_sum_second_order_x2_x_kuu
#



# @jit
# def heat_equation_kfu(x1, x2, params) -> jnp.ndarray:
#     total_sum_second_order_x1_x_kuu = jnp.zeros_like(deep_second_order_x1(x1, x2, params, 0))
#     num_x1 = x1.shape[1] - 1
#     for d in range(num_x1):
#         total_sum_second_order_x1_x_kuu += deep_second_order_x1(x1, x2, params, d)
#     return deep_first_order_x1_t_kuu(x1, x2, params) - total_sum_second_order_x1_x_kuu
#
#
#
#
#
# @jit
# def deep_first_order_x1_t_kuu(x1: ArrayLike, x2: ArrayLike, params) -> jnp.ndarray:
#     return deep_first_order_x1(x1, x2, params, -1)

#
# @jit
# def deep_second_order_x1(x1, x2, params, init_d):
#     d = init_d
#     v = jnp.ones(len(x1[:, d]))
#
#     def extract_fn(x1_col):
#         x1_mod = x1.at[:, d].set(x1_col)
#         return deep_first_order_x1(x1_mod, x2, params, d)
#
#     _, jvp_fn = jvp(extract_fn, (x1[:, d],), (v,))
#     return jvp_fn
#
#
# @jit
# def deep_first_order_x1(x1: ArrayLike, x2: ArrayLike, params, init_d) -> jnp.ndarray:
#     d = init_d
#
#     def kernel_fn(x1_col):
#         x1_mod = x1.at[:, d].set(x1_col)
#         return deep_rbf_kernel(x1_mod, x2, params)
#
#     v = jnp.ones(len(x1[:, d]))
#     _, jvp_fn = jvp(kernel_fn, (x1[:, d],), (v,))
#     return jvp_fn
#
#
# @jit
# def deep_first_order_x1_t_kuf(x1: ArrayLike, x2: ArrayLike, params) -> jnp.ndarray:
#     d = -1
#     return deep_first_order_x1_x_kuf(x1, x2, params, d)
#
#
# @jit
# def deep_first_order_x1_x_kuf(x1: ArrayLike, x2: ArrayLike, params, init_d) -> jnp.ndarray:
#     d = init_d
#
#     def kernel_fn(x1_col):
#         x1_mod = x1.at[:, d].set(x1_col)
#         return deep_heat_equation_high_d_kuf(x1_mod, x2, params)
#
#     v = jnp.ones(len(x1[:, d]))
#     _, jvp_fn = jvp(kernel_fn, (x1[:, d],), (v,))
#     return jvp_fn
#
#
# @jit
# def deep_heat_equation_high_d_kuf(x1: ArrayLike, x2: ArrayLike, params) -> jnp.ndarray:
#     total_sum_second_order_x2_x_kuu = jnp.zeros_like(deep_second_order_x2_x_kuu(x1, x2, params, 0))
#     num_x2 = x2.shape[1] - 1
#     for d in range(num_x2):
#         total_sum_second_order_x2_x_kuu += deep_second_order_x2_x_kuu(x1, x2, params, d)
#     return deep_first_order_x2_t_kuu(x1, x2, params) - total_sum_second_order_x2_x_kuu
#
#
# @jit
# def deep_first_order_x2(x1: ArrayLike, x2: ArrayLike, params, init_d) -> jnp.ndarray:
#     d = init_d
#
#     def kernel_fn(x2_col):
#         x2_mod = x2.at[:, d].set(x2_col)
#         return deep_rbf_kernel(x1, x2_mod, params)
#
#     v = jnp.ones(len(x2[:, d]))
#     _, jvp_fn = jvp(kernel_fn, (x2[:, d],), (v,))
#     return jvp_fn
#
#
#
#
#
# @jit
# def deep_first_order_x2_t_kuu(x1: ArrayLike, x2: ArrayLike, params) -> jnp.ndarray:
#     return deep_first_order_x2(x1, x2, params, -1)
#
#
# @jit
# def deep_second_order_x2_x_kuu(x1: ArrayLike, x2: ArrayLike, params, init_d) -> jnp.ndarray:
#     d = init_d
#     v = jnp.ones(len(x2[:, d]))
#
#     def extract_fn(x2_col):
#         x2_mod = x2.at[:, d].set(x2_col)
#         return deep_first_order_x2(x1, x2_mod, params, d)
#
#     _, jvp_fn = jvp(extract_fn, (x2[:, d],), (v,))
#     return jvp_fn
#
#
# @jit
# def deep_second_order_x1_x_kuf(x1: ArrayLike, x2: ArrayLike, params, init_d) -> jnp.ndarray:
#     d = init_d
#     v = jnp.ones(len(x1[:, d]))
#
#     def extract_fn(x1_col):
#         x1_mod = x1.at[:, d].set(x1_col)
#         return deep_first_order_x1_x_kuf(x1_mod, x2, params, d)
#
#     _, jvp_fn = jvp(extract_fn, (x1[:, d],), (v,))
#     return jvp_fn