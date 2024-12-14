import datetime

import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from jax import jit
from jax.typing import ArrayLike
from math import pi
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import qmc

plt.rcParams.update({"figure.figsize": (12, 6)})
plt.rcParams.update({'font.size': 22})

def f_xt(Xf) -> jnp.ndarray:
    x = Xf[:, :1]
    t = Xf[:, -1:]

    f: jnp.ndarray = jnp.exp((-1) * t) * (16 * (pi ** 2) + (-1)) * jnp.sin(4 * pi * x)
    return f


def u_xt(Xu_fixed) -> jnp.ndarray:
    x = Xu_fixed[:, :1]
    t = Xu_fixed[:, -1:]

    u: jnp.ndarray = jnp.exp((-1) * t) * jnp.sin(4 * pi * x)
    return u

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


@jit
def compute_kuu_rd(x1: ArrayLike, x2: ArrayLike, params) -> jnp.ndarray:
    return deep_rbf_kernel(x1, x2, params)


@jit
def compute_kuf_rd(x1: jnp.ndarray, x2: jnp.ndarray, initial_theta, lengthscale_x, lengthscale_t) -> jnp.ndarray:
    t1 = x1[:, -1]
    t2 = x2[:, -1]
    x1_spatial = x1[:, :-1]
    x2_spatial = x2[:, :-1]

    params = {'sigma': initial_theta[-1][0], 'lengthscale': initial_theta[-1][1]}
    k = deep_rbf_kernel(x1, x2, params)

    term1 = (t1[:, None] - t2[None, :]) / lengthscale_t ** 2
    term2 = 0.01 * (jnp.sum((x1_spatial[:, None] - x2_spatial[None, :]) ** 2, axis=-1) / lengthscale_x ** 4)
    term3 = 0.01 *  (1 / lengthscale_x ** 2)

    k_uf = k * (term1 + term3 - term2)

    return k_uf

@jit
def compute_kfu_rd(x1: jnp.ndarray, x2: jnp.ndarray, initial_theta, lengthscale_x, lengthscale_t) -> jnp.ndarray:
    t1 = x1[:, -1]
    t2 = x2[:, -1]
    x1_spatial = x1[:, :-1]
    x2_spatial = x2[:, :-1]

    params = {'sigma': initial_theta[-1][0], 'lengthscale': initial_theta[-1][1]}
    k = deep_rbf_kernel(x1, x2, params)

    term1 = -(t1[:, None] - t2[None, :]) / lengthscale_t ** 2
    term2 = 0.01 * (jnp.sum((x2_spatial[None] - x1_spatial[:, None]) ** 2, axis=-1) / lengthscale_x ** 4)
    term3 = 0.01 *  (1 / lengthscale_x ** 2)

    k_fu = k * (term1 - term2 + term3)

    return k_fu


@jit
def compute_kff_rd(x1: jnp.ndarray, x2: jnp.ndarray, initial_theta, lengthscale_x, lengthscale_t) -> jnp.ndarray:
    t1 = x1[:, -1]
    t2 = x2[:, -1]
    x1_spatial = x1[:, :-1]
    x2_spatial = x2[:, :-1]

    params = {'sigma': initial_theta[-1][0], 'lengthscale': initial_theta[-1][1]}
    k = deep_rbf_kernel(x1, x2, params)

    alpha = 0.01
    term1 = k * ((t1[:, None] - t2) / lengthscale_t ** 2) * (
            -(t1[:, None] - t2) / lengthscale_t ** 2
            - (alpha * (jnp.sum((x2_spatial[None] - x1_spatial[:, None]) ** 2, axis=-1) / lengthscale_x ** 4))
            + (alpha * (1 / lengthscale_x ** 2))
    ) + k / lengthscale_t ** 2

    term2 = (2 * alpha * k / lengthscale_x ** 6) * jnp.sum((x1_spatial[:, None] - x2_spatial[None]) ** 2, axis=-1) - (
                2 * alpha * k / lengthscale_x ** 4)

    term3 = -k * (t2 - t1[:, None]) / (lengthscale_x ** 2 * lengthscale_t ** 2) + k * jnp.sum(
        (x1_spatial[:, None] - x2_spatial[None]) ** 2, axis=-1) * (t2 - t1[:, None]) / (
                        lengthscale_x ** 4 * lengthscale_t ** 2)

    term4 = -alpha * k / lengthscale_x ** 4 + alpha * (
                k * jnp.sum((x1_spatial[:, None] - x2_spatial[None]) ** 2, axis=-1) / lengthscale_x ** 6)

    term5 = k * (3 * alpha * jnp.sum((x2_spatial[None] - x1_spatial[:, None]) ** 2, axis=-1) / lengthscale_x ** 6 - (
                alpha * jnp.sum((x2_spatial[None] - x1_spatial[:, None]) ** 4, axis=-1) / lengthscale_x ** 8))

    total_expression = term1 - alpha * (term2 + term3 + term4 + term5)

    return total_expression


def heat_equation_nlml_loss_2d_rd_no(heat_params, Xfz, Xfg, number_Y, Y) -> float:
    init = heat_params
    params =  heat_params
    params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    lengthscale_x = params[0][1][0].item()
    lengthscale_t = params[0][1][1].item()

    zz_ff = compute_kuu_rd(Xfz, Xfz, params_kuu)
    zg_ff = compute_kuf_rd(Xfz, Xfg, params, lengthscale_x, lengthscale_t)
    gz_ff = compute_kfu_rd(Xfg, Xfz, params, lengthscale_x, lengthscale_t)
    gg_ff = compute_kff_rd(Xfg, Xfg, params, lengthscale_x, lengthscale_t)

    K = jnp.block([[zz_ff, zg_ff], [gz_ff, gg_ff]])

    sign, logdet = jnp.linalg.slogdet(K)
    K_inv_Y = jnp.linalg.solve(K, Y)
    signed_logdet = sign * logdet
    K_inv_Y_product = Y.T @ K_inv_Y
    scalar_result = jnp.squeeze(K_inv_Y_product)
    nlml = (1 / 2 * signed_logdet) + (1 / 2 * scalar_result) + ((number_Y / 2) * jnp.log(2 * jnp.pi))

    return  nlml


def heat_equation_nlml_loss_2d_rd(heat_params, Xuz, Xfz, Xfg, number_Y, Y) -> float:

    init = heat_params
    params =  heat_params
    params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    lengthscale_x = params[0][1][0].item()
    lengthscale_t = params[0][1][1].item()

    zz_uu = compute_kuu_rd(Xuz, Xuz, params_kuu)
    zz_uf = compute_kuu_rd(Xuz, Xfz, params_kuu)
    zg_uf = compute_kuf_rd(Xuz, Xfg, params, lengthscale_x, lengthscale_t)
    zz_fu = compute_kuu_rd(Xfz, Xuz, params_kuu)
    zz_ff = compute_kuu_rd(Xfz, Xfz, params_kuu)
    zg_ff = compute_kuf_rd(Xfz, Xfg, params, lengthscale_x, lengthscale_t)
    gz_fu = compute_kfu_rd(Xfg, Xuz, params, lengthscale_x, lengthscale_t)
    gz_ff = compute_kfu_rd(Xfg, Xfz, params, lengthscale_x, lengthscale_t)
    gg_ff = compute_kff_rd(Xfg, Xfg, params, lengthscale_x, lengthscale_t)

    K = jnp.block([[zz_uu, zz_uf, zg_uf], [zz_fu, zz_ff, zg_ff], [gz_fu, gz_ff, gg_ff]])

    sign, logdet = jnp.linalg.slogdet(K)
    K_inv_Y = jnp.linalg.solve(K, Y)
    signed_logdet = sign * logdet
    K_inv_Y_product = Y.T @ K_inv_Y
    scalar_result = jnp.squeeze(K_inv_Y_product)
    nlml = (1 / 2 * signed_logdet) + (1 / 2 * scalar_result) + ((number_Y / 2) * jnp.log(2 * jnp.pi))

    return  nlml

def exp_nested_tuple(nested_tuple):
    if isinstance(nested_tuple, tuple):
        return tuple(exp_nested_tuple(elem) for elem in nested_tuple)
    elif isinstance(nested_tuple, jnp.ndarray):
        return jnp.exp(nested_tuple)
    else:
        return nested_tuple

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

    K = jnp.block([[zz_uu, zz_uf, zg_uf], [zz_fu, zz_ff, zg_ff], [gz_fu, gz_ff, gg_ff]])

    sign, logdet = jnp.linalg.slogdet(K)
    K_inv_Y = jnp.linalg.solve(K, Y)
    signed_logdet = sign * logdet
    K_inv_Y_product = Y.T @ K_inv_Y
    scalar_result = jnp.squeeze(K_inv_Y_product)
    nlml = (1 / 2 * signed_logdet) + (1 / 2 * scalar_result) + ((number_Y / 2) * jnp.log(2 * jnp.pi))

    return  nlml

def plot_u_f(Xu_certain_all, Xf, Xu_noise, noise_std):
    x = jnp.linspace(0, 1, 100)
    t = jnp.linspace(0, 1, 100)
    X, T = jnp.meshgrid(x, t)
    X_plot = jnp.vstack([X.ravel(), T.ravel()]).T

    f_values = f_xt(X_plot).reshape(X.shape)
    u_values = u_xt(X_plot).reshape(X.shape)


    fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)


    c1 = ax[0].contourf(X, T, f_values, levels=50, cmap='viridis')
    fig.colorbar(c1, ax=ax[0], orientation='vertical', label='f(x, t) value')
    ax[0].scatter(Xf[:, 0], Xf[:, 1], color='black', label='Xu points')
    ax[0].set_title('f(x, t)', fontsize=16)
    ax[0].set_xlabel('x', fontsize=14)
    ax[0].set_ylabel('t', fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=12)


    c2 = ax[1].contourf(X, T, u_values, levels=50, cmap='plasma')
    fig.colorbar(c2, ax=ax[1], orientation='vertical', label='u(x, t) value')
    ax[1].scatter(Xu_certain_all[:, 0], Xu_certain_all[:, 1], color='black', label='Xu points')
    ax[1].scatter(Xu_noise[:, 0], Xu_noise[:, 1], color='blue', label='Xu noise points', marker='o')
    ax[1].set_title('u(x, t)', fontsize=16)
    ax[1].set_xlabel('x', fontsize=14)
    ax[1].set_ylabel('t', fontsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=12)


    for a in ax:
        a.set_aspect('auto')
        a.grid(True, linestyle='--', alpha=0.6)
    current_time = datetime.datetime.now().strftime("%M%S")
    plt.suptitle('Function Plots of f(x, t) and u(x, t)', fontsize=18)
    plt.savefig(f'u_f_plot_{noise_std}_{current_time}.png')


def plot_u_f_pred(Xu_certain_all, Xf, Xu_noise, noise_std, Xu_pred, prior_var,assumption_sigma,k,max_samples):
    x = jnp.linspace(0, 1, 100)
    t = jnp.linspace(0, 1, 100)
    X, T = jnp.meshgrid(x, t)
    X_plot = jnp.vstack([X.ravel(), T.ravel()]).T

    f_values = f_xt(X_plot).reshape(X.shape)
    u_values = u_xt(X_plot).reshape(X.shape)

    num_f = Xf.shape[0]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)


    c1 = ax[0].contourf(X, T, f_values, levels=50, cmap='viridis')
    fig.colorbar(c1, ax=ax[0], orientation='vertical', label='f(x, t) value')
    ax[0].scatter(Xf[:, 0], Xf[:, 1], color='black', label='Xu points')
    ax[0].set_title('f(x, t)', fontsize=16)
    ax[0].set_xlabel('x', fontsize=14)
    ax[0].set_ylabel('t', fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=12)


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

    plt.savefig(f'pred_k{k}_priorvar_{prior_var}_assump{assumption_sigma}_nstd{noise_std}_iter{max_samples}_{current_time}.png')


def plot_u_pred(Xu_certain_all, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred, prior_var,assumption_sigma,k,max_samples,learning,num_chains,number_f,added_text):
    x = jnp.linspace(0, 1, 100)
    t = jnp.linspace(0, 1, 100)
    X, T = jnp.meshgrid(x, t)
    X_plot = jnp.vstack([X.ravel(), T.ravel()]).T
    u_values = u_xt(X_plot).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)


    c1 = ax.contourf(X, T, u_values, levels=50, cmap='plasma')

    fig.colorbar(c1, ax=ax, orientation='vertical', label='u(x, t) value')

    ax.scatter(Xu_certain[:, 0], Xu_certain[:, 1], color='black', label='GT', marker='o')
    ax.scatter(Xu_noise[0:4, 0], Xu_noise[0:4, 1], color='tab:blue', label='Xu noise inner', marker='x')
    ax.scatter(Xu_noise[4:, 0], Xu_noise[4:, 1], color='tab:orange', label='Xu noise t', marker='x')
    ax.scatter(Xu_pred[:, 0], Xu_pred[:, 1], color='tab:red', label='Posterior', marker='o')
    ax.set_title('u(x, t)', fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('t', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_aspect('auto')
    ax.grid(True, linestyle='--', alpha=0.6)

    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f'contourf_{added_text}.png')

def get_u_value_from_data(x, t, data):
    Ny, Nx = data.shape
    x_idx = int(round((x + 1) / 2 * (Nx - 1)))
    x_idx = jnp.clip(x_idx, 0, Nx-1)

    t_idx = int(round((1 - t) * (Ny - 1)))
    t_idx = jnp.clip(t_idx, 0, Ny-1)

    return data[t_idx, x_idx]


def plot_u_pred_rd(Xu_certain, Xf, Xu_noise, noise_std, Xu_pred, prior_var,assumption_sigma,k,max_samples,learning,num_chains,number_f,added_text, X_plot_prediction, data):
    X_plot = X_plot_prediction
    X = X_plot[:, 0]
    T = X_plot[:, 1]
    u_values = data

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    c1 =ax.imshow(data, extent=[-1,1,1,0])

    fig.colorbar(c1, ax=ax, orientation='vertical')

    ax.scatter(Xu_certain[:, 0], Xu_certain[:, 1], color='black', label='GT', marker='o')
    ax.scatter(Xu_noise[:, 0], Xu_noise[:, 1], color='tab:blue', label='Xu noise', marker='x')
    ax.scatter(Xu_pred[:, 0], Xu_pred[:, 1], color='tab:red', label='Posterior', marker='o')
    ax.set_title('u(x, t)', fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('t', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_aspect('auto')
    ax.grid(True, linestyle='--', alpha=0.6)

    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f'rd_contourf_{added_text}.png')


def plot_u_pred_rd_value(Xu_certain, Xf, Xu_noise, noise_std, Xu_pred, prior_var,
                   assumption_sigma, k, max_samples, learning, num_chains,
                   number_f, added_text, X_plot_prediction, data):
    X_plot = X_plot_prediction
    X = X_plot[:, 0]
    T = X_plot[:, 1]
    u_values = data

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)


    c1 = ax.imshow(data, extent=[-1,1,1,0])
    fig.colorbar(c1, ax=ax, orientation='vertical')

    ratios = []
    for (x_pred, t_pred), (x_gt, t_gt) in zip(Xu_noise, Xu_certain):
        u_pred_val = get_u_value_from_data(x_pred, t_pred, data)
        u_gt_val = get_u_value_from_data(x_gt, t_gt, data)
        dist = np.sqrt((x_pred - x_gt) ** 2 + (t_pred - t_gt) ** 2)
        if dist == 0:
            ratio = 0.0
        else:
            ratio = abs(u_pred_val - u_gt_val) / dist
        ratios.append(ratio)

    ratios = jnp.array(ratios)
    norm = plt.Normalize(vmin=ratios.min(), vmax=ratios.max())
    cmap = plt.cm.Reds

    for i, (x_pred, t_pred) in enumerate(Xu_noise):
        ax.annotate(f"{ratios[i]:.2f}",
                    xy=(x_pred, t_pred), xycoords='data',
                    xytext=(x_pred + 0.04, t_pred + 0.04), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color='black'),
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    ax.scatter(Xu_certain[:, 0], Xu_certain[:, 1], color='black', label='GT', marker='o')
    ax.scatter(Xu_noise[:, 0], Xu_noise[:, 1], color='tab:blue', label='Xu noise', marker='x')
    ax.scatter(Xu_pred[:, 0], Xu_pred[:, 1], color='tab:red', label='Posterior', marker='o')

    ax.set_title('u(x, t)', fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('t', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ratio_label = mpatches.Patch(color='white', alpha=0.8, label='ratio')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(ratio_label)
    labels.append("ratio")
    ax.legend(handles=handles, labels=labels, loc='best')
    ax.set_aspect('auto')
    ax.grid(True, linestyle='--', alpha=0.6)



    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f'rd_contourf_value_{added_text}.png')


def plot_u_pred_rd_value_2(Xu_certain, Xf, Xu_noise, noise_std, Xu_pred, prior_var,
                   assumption_sigma, k, max_samples, learning, num_chains,
                   number_f, added_text, X_plot_prediction, data):
    Ny, Nx = data.shape
    t_array = jnp.linspace(0, 1, Ny)
    x_array = jnp.linspace(-1, 1, Nx)
    data_flip = jnp.flipud(data)

    interp_func = RegularGridInterpolator((t_array, x_array), data_flip, bounds_error=False, fill_value=None)


    def get_u_value_from_data(x, t):

        t_new = 1 - t

        return interp_func((t_new, x))

    X_plot = X_plot_prediction
    X = X_plot[:, 0]
    T = X_plot[:, 1]
    u_values = data

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    c1 = ax.imshow(data, extent=[-1, 1, 1, 0])
    fig.colorbar(c1, ax=ax, orientation='vertical')

    ratios = []
    for (x_pred, t_pred), (x_gt, t_gt) in zip(Xu_pred, Xu_certain):
        u_pred_val = get_u_value_from_data(x_pred, t_pred)
        u_gt_val = get_u_value_from_data(x_gt, t_gt)
        dist = jnp.sqrt((x_pred - x_gt) ** 2 + (t_pred - t_gt) ** 2)
        if dist == 0:
            ratio = 0.0
        else:
            ratio = abs(u_pred_val - u_gt_val) / dist
        ratios.append(ratio)

    ratios = jnp.array(ratios)
    norm = plt.Normalize(vmin=ratios.min(), vmax=ratios.max())
    cmap = plt.cm.Reds

    for i, (x_pred, t_pred) in enumerate(Xu_pred):
        color = cmap(norm(ratios[i]))
        ax.text(x_pred + 0.1, t_pred + 0.01, f"{ratios[i]:.2f}",
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='square,pad=0.2', facecolor=color, edgecolor='none', alpha=0.7), zorder=1)

    ax.scatter(Xu_certain[:, 0], Xu_certain[:, 1], color='black', label='GT', marker='o', zorder=2)
    ax.scatter(Xu_noise[:, 0], Xu_noise[:, 1], color='tab:blue', label='Xu noise', marker='x', linewidths=2, zorder=3)
    ax.scatter(Xu_pred[:, 0], Xu_pred[:, 1], color='tab:red', label='Posterior', marker='o', zorder=4)

    ax.set_title('u(x, t)', fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('t', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ratio_label = mpatches.Patch(color='white', alpha=0.8, label='ratio')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(ratio_label)
    labels.append("ratio")
    ax.legend(handles=handles, labels=labels, loc='best')
    ax.set_aspect('auto')
    ax.grid(True, linestyle='--', alpha=0.6)

    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f'rd_contourf_value2_{added_text}.png')



def plot_u_pred_rd_value_3(Xu_certain, Xf, Xu_noise, noise_std, Xu_pred, prior_var,
                   assumption_sigma, k, max_samples, learning, num_chains,
                   number_f, added_text, X_plot_prediction, data):
    Ny, Nx = data.shape
    t_array = jnp.linspace(0, 1, Ny)
    x_array = jnp.linspace(-1, 1, Nx)
    data_flip = jnp.flipud(data)

    interp_func = RegularGridInterpolator((t_array, x_array), data_flip, bounds_error=False, fill_value=None)


    def get_u_value_from_data(x, t):

        t_new = 1 - t

        return interp_func((t_new, x))

    X_plot = X_plot_prediction
    X = X_plot[:, 0]
    T = X_plot[:, 1]
    u_values = data

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    c1 = ax.imshow(data, extent=[-1, 1, 1, 0])
    fig.colorbar(c1, ax=ax, orientation='vertical')

    ratios = []
    for (x_prior, t_prior), (x_gt, t_gt) in zip(Xu_noise, Xu_certain):
        u_prior_val = get_u_value_from_data(x_prior, t_prior)
        u_gt_val = get_u_value_from_data(x_gt, t_gt)
        dist = jnp.sqrt((x_prior - x_gt) ** 2 + (t_prior - t_gt) ** 2)
        if dist == 0:
            ratio = 0.0
        else:
            ratio = abs(u_prior_val - u_gt_val) / dist
        ratios.append(ratio)

    ratios = jnp.array(ratios)
    norm = plt.Normalize(vmin=ratios.min(), vmax=ratios.max())
    cmap = plt.cm.Reds

    for i, (x_prior, t_prior) in enumerate(Xu_noise):
        color = cmap(norm(ratios[i]))
        ax.text(x_prior + 0.1, t_prior + 0.01, f"{ratios[i]:.2f}",
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='square,pad=0.2', facecolor=color, edgecolor='none', alpha=0.7), zorder=1)


    ax.scatter(Xu_certain[:, 0], Xu_certain[:, 1], color='black', label='GT', marker='o', zorder=2)
    ax.scatter(Xu_noise[:, 0], Xu_noise[:, 1], color='tab:blue', label='Xu noise', marker='x', linewidths=2, zorder=3)
    ax.scatter(Xu_pred[:, 0], Xu_pred[:, 1], color='tab:red', label='Posterior', marker='o', zorder=4)

    ax.set_title('u(x, t)', fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('t', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ratio_label = mpatches.Patch(color='white', alpha=0.8, label='ratio')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(ratio_label)
    labels.append("ratio")
    ax.legend(handles=handles, labels=labels, loc='best')
    ax.set_aspect('auto')
    ax.grid(True, linestyle='--', alpha=0.6)

    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f'rd_contourf_value3_{added_text}.png')


def get_u_training_data_2d(key_x_u, key_x_u_init, key_t_u_low, key_t_u_high, key_x_noise, key_t_noise, sample_num,
                           init_num, bnum, noise_std) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                              jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray):
    Xu_certain = jax.random.uniform(key_x_u, shape=(sample_num, 2), dtype=jnp.float64)
    yu_certain = u_xt(Xu_certain)

    xu, tu = Xu_certain[:, :1], Xu_certain[:, -1:]

    xu_noise = xu -jnp.abs(noise_std * jax.random.normal(key_x_noise, shape=xu.shape))
    tu_noise = tu -jnp.abs(noise_std * jax.random.normal(key_t_noise, shape=tu.shape))

    Xu_noise = jnp.concatenate([xu_noise, tu_noise], axis=1)

    xu_init = jax.random.uniform(key_x_u_init, shape=(init_num, 1), dtype=jnp.float64)
    tu_init = jnp.zeros(shape=(init_num, 1))

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
    sample_num_total = sample_num + number_u_only_x
    Xu_certain = jax.random.uniform(key_x_u, shape=(sample_num_total, 2), dtype=jnp.float64)

    Xu_certain = jnp.maximum(0, jnp.minimum(1, Xu_certain))

    xu, tu = Xu_certain[:, :1], Xu_certain[:, -1:]
    yu_certain = u_xt(Xu_certain)

    xu_noise = xu + noise_std * jax.random.normal(key_x_noise, shape=xu.shape)
    tu_with_noise = tu[:sample_num] + noise_std * jax.random.normal(key_t_noise, shape=tu[:sample_num].shape)

    tu_noise = jnp.concatenate([tu_with_noise, tu[sample_num:]])
    Xu_noise = jnp.concatenate([xu_noise, tu_noise], axis=1)
    Xu_noise = jnp.maximum(0, jnp.minimum(1, Xu_noise))

    print("Xu_noise: ", Xu_noise)
    print("tu: ", tu)
    print("tu_noise: ", tu_noise)
    print("xu_noise: ", xu_noise)
    print("xu: ", xu)

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

    xu_fixed = jnp.concatenate((xu_bound_low, xu_init, xu_bound_high,))
    tu_fixed = jnp.concatenate((tu_bound_low, tu_init, tu_bound_high))
    Xu_fixed = jnp.concatenate([xu_fixed, tu_fixed], axis=1)

    Yu_fixed = u_xt(Xu_fixed)


    return Xu_certain, yu_certain, xu_noise, tu_noise, Xu_noise, xu_fixed, tu_fixed, Xu_fixed, Yu_fixed, init_num, bnum


def get_f_training_data_2d(key_x_f, sample_num) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray):
    xf = (jnp.cos((((2 * jnp.arange(sample_num)) + 1) / (2 * (sample_num))) * jnp.pi) + 1) / 2
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
