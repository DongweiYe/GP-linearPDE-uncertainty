import jax
import datetime
import optax
from include.config import key_x_rd, key_t_rd
from include.config import key_num
from include.heat2d import plot_u_pred, plot_u_pred_rd
from include.init import initialize_params_2d
from include.mcmc_posterior import *

from include.plot_dist import plot_dist, plot_with_noise, plot_and_save_kde_histograms, plot_dist_rd, plot_with_noise_rd
from include.train import train_heat_equation_model_2d, train_heat_equation_model_2d_rd
from scipy.stats import gaussian_kde
import pickle
import os

import numpy as np
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from skfdiff import Model, Simulation
from scipy.interpolate import CubicSpline
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import griddata

from test_f_infer_function import plot_f_inference_rd

os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_enable_x64", True)


# %%
learning_rate = 0.04
epochs = 1000 #1000

noise_std = 0.04
prior_std = 0.04
prior_var = prior_std ** 2  # prior variance
max_samples = 1000
assumption_sigma = 0.01 #0.01 # step size
k = 0.8
num_chains = 1

bw = 2
num_prior_samples = 200

test_num = 2 ** 4
number_u = 2 ** 2  # xtxx
number_init = 2 ** 5
number_bound = 2 ** 5
number_u_c_for_f = 2 ** 4

number_u_c_for_f_real = (number_u_c_for_f)**2
number_init_real = number_init-1
number_bound_real = (number_bound+1)*2
number_f = number_u_c_for_f_real+number_init_real+number_bound_real

init_num = number_init
bnum = number_bound
keynum = 50
optimizer_in_use = optax.adam
learning_rate_pred = 0.01
epoch_pred = 100
added_text = f'key{keynum}_{number_u}&{number_u_c_for_f_real}&{number_f}&{number_init}&{number_bound}&{epochs}&{noise_std}'
learning = f'lr{learning_rate}&{epochs}'
mcmc_text = f"key{keynum}_number_u_c_for_f{number_u_c_for_f}noise{noise_std}_prior{prior_std}_maxsamples{max_samples}_assumption{assumption_sigma}_k{k}"


# %%
if __name__ == '__main__':
    print("noise_std:", noise_std, "\n")
    print("prior_var:", prior_var, "\n")
    print("number_u:", number_u, "\n")
    print("number_f:", number_f, "\n")
    print("number_init:", number_init, "\n")
    print("number_bound:", number_bound, "\n")
    print("max_samples:", max_samples, "\n")
    print("assumption_sigma:", assumption_sigma, "\n")
    print("k:", k, "\n")
    print("optimizer_in_use:", optimizer_in_use, "\n")
    print("epochs:", epochs, "\n")
    print("added_text:", added_text, "\n")
    print("learning_rate:", learning_rate, "\n")


    model = Model("k * (dxxT) - 3*T**3 + 3*T", "T(x)", parameters="k", boundary_conditions="periodic", backend='numpy')
    # model = Model("k * (dxxT) - 3*T**3", "T(x)", parameters="k", boundary_conditions="periodic", backend='numpy')
    x = jnp.linspace(-1, 1, 500)
    T = x * x * jnp.cos(jnp.pi * x)
    initial_fields = model.Fields(x=x, T=T, k=0.05)

    simulation = Simulation(model, initial_fields, dt=0.002, tmax=1, scheme="theta")

    data = [T]
    for t, fields in simulation:
        data.append(fields.T)

    data = jnp.asarray(data)
    timesteps, spatial_points = data.shape

    time_grid = jnp.linspace(0, 1, timesteps)
    x_grid = x
    x_grid_mesh, time_grid_mesh = jnp.meshgrid(x_grid, time_grid)
    x_grid_mesh_shape = x_grid_mesh.shape

    num_initial_samples = number_init
    num_boundary_samples = number_bound
    num_samples = number_u_c_for_f

    # xu_init = (jnp.cos(jnp.arange(init_num + 1) * jnp.pi / init_num))
    # xu_init = xu_init[1:-1, ]
    # xu_init = jnp.expand_dims(xu_init, axis=-1)
    # print("init_num prior", init_num)
    # init_num = xu_init.shape[0]
    # print("init num after", init_num)
    # tu_init = jnp.zeros(shape=(init_num, 1))
    #
    # Yu_init = RBFInterpolator(jnp.expand_dims(x_grid, axis=1), data[0:1, :].T)(xu_init)
    # Yu_init = jnp.squeeze(Yu_init)
    # Xu_init = jnp.hstack([xu_init, tu_init])
    #
    # tu_bound_low = ((jnp.cos(jnp.arange(bnum + 1) * jnp.pi / bnum)) + 1) / 2
    # tu_bound_low = jnp.expand_dims(tu_bound_low, axis=-1)
    # tu_bound_high = ((jnp.cos(jnp.arange(bnum + 1) * jnp.pi / bnum)) + 1) / 2
    # tu_bound_high = jnp.expand_dims(tu_bound_high, axis=-1)
    # print("bnum before", bnum)
    # bnum = tu_bound_low.shape[0]
    # print("bnum after", bnum)
    # xu_bound_low = -jnp.ones(shape=(bnum, 1))
    # xu_bound_high = jnp.ones(shape=(bnum, 1))
    #
    # Yu_bound_low = RBFInterpolator(jnp.expand_dims(time_grid, axis=1), data[:, 0:1])(tu_bound_low)
    # Yu_bound_low = jnp.squeeze(Yu_bound_low)
    # Xu_bound_low = jnp.hstack([xu_bound_low, tu_bound_low])
    #
    # Yu_bound_high = RBFInterpolator(jnp.expand_dims(time_grid, axis=1), data[:, -1:])(tu_bound_high)
    # Yu_bound_high = jnp.squeeze(Yu_bound_high)
    # Xu_bound_high = jnp.hstack([xu_bound_high, tu_bound_high])
    # print("before num_samples", num_samples)

    xu_all = jnp.cos((((2 * jnp.arange(num_samples)) + 1) / (2 * (num_samples))) * jnp.pi)
    tu_all = (jnp.cos((((2 * jnp.arange(num_samples)) + 1) / (2 * (num_samples))) * jnp.pi) + 1) / 2

    # xu_all = jax.random.uniform(key_x_rd, shape=(num_samples, 1), dtype=jnp.float64)
    # tu_all = jax.random.uniform(key_t_rd, shape=(num_samples, 1), dtype=jnp.float64)

    # epsilon = 1e-6
    # xu_all = jnp.linspace(-1 + epsilon, 1 - epsilon, num_samples, dtype=jnp.float64)
    # tu_all = jnp.linspace(0 + epsilon, 1 - epsilon, num_samples, dtype=jnp.float64)

    Xu_all = jnp.vstack([xu_all, tu_all]).T
    Xu_mesh_all, Tu_mesh_all = jnp.meshgrid(xu_all, tu_all)
    Xu_inner_all = Xu_mesh_all.ravel()
    Tu_inner_all = Tu_mesh_all.ravel()
    num_sample = Xu_inner_all.shape
    print("num_sample num after", num_sample)
    U_inner_all = jnp.vstack([Xu_inner_all, Tu_inner_all]).T


    key_u_rd = random.PRNGKey(keynum)
    print("keynum:", keynum)
    random_time_indices_internal = random.randint(key_u_rd, (number_u,), 1, timesteps - 1)
    random_space_indices_internal = random.randint(key_u_rd, (number_u,), 0, spatial_points)
    x_u = x_grid[random_space_indices_internal]
    t_u = time_grid[random_time_indices_internal]
    X_u = jnp.vstack([x_u, t_u]).T

    train_mesh_points = jnp.vstack([time_grid_mesh.ravel(), x_grid_mesh.ravel()]).T

    inner_query_points = jnp.vstack([Tu_inner_all.ravel(), Xu_inner_all.ravel()]).T
    Yu_inner_all = griddata(
        train_mesh_points,
        data.flatten(),
        inner_query_points,
        method='cubic'
    )

    u_query_points = jnp.vstack([t_u.ravel(), x_u.ravel()]).T
    yu = griddata(
        train_mesh_points,
        data.flatten(),
        u_query_points,
        method='cubic'
    )

    Xu_plot = jnp.vstack([U_inner_all, X_u])  #
    yu_plot = jnp.concatenate([Yu_inner_all, yu])

    Xu_fixed= jnp.vstack([U_inner_all])  #
    yu_fixed = jnp.concatenate([Yu_inner_all])

    plt.scatter(Tu_mesh_all, Xu_mesh_all, c="red")
    # plt.scatter(tu_init, xu_init, c="blue")
    # plt.scatter(tu_bound_low, xu_bound_low, c="green")
    # plt.scatter(tu_bound_high, xu_bound_high, c="green")
    plt.scatter(t_u, x_u, color="purple")
    plt.show()
    print("Xu_plot shape", Xu_plot.shape)
    print("Yu_plot shape", yu_plot.shape)
    # plt.savefig('data_with_points.png')

    Xu_certain = X_u
    yu_certain = yu
    key_x_noise, key_t_noise = random.split(key_u_rd)
    xu_noise = x_u + noise_std * jax.random.normal(key_x_noise, shape=x_u.shape)
    tu_noise = t_u + noise_std * jax.random.normal(key_t_noise, shape=t_u.shape)
    Xu_noise = jnp.vstack([xu_noise, tu_noise]).T

    print("Xu_certain:", Xu_certain)
    print("Xu_noise:", Xu_noise)
    print("yu_certain:", yu_certain)

    print("Xu_fixed:", Xu_fixed)
    print("Yu_fixed:", yu_fixed)

    Yu = jnp.concatenate((yu_certain, yu_fixed))
    Xu_all_with_noise = jnp.concatenate((Xu_noise, Xu_fixed))
    Xu_all_without_noise = jnp.concatenate((Xu_certain, Xu_fixed))

    print("Xu_all_with_noise:", Xu_all_with_noise)
    print("Xu_all_without_noise:", Xu_all_without_noise)
    print("Yu:", Yu)

    # -Lu = R(u)
    beta = 3
    R_u = beta * (yu_fixed**3 - yu_fixed)
    # R_u = beta * (yu_fixed**3)
    Lu_data = -R_u

    Xf = Xu_fixed
    yf = Lu_data

    print("Xf (chosen points):", Xf)
    print("yf (corresponding Lu values):", yf)

    Y_certain = jnp.concatenate((yu_fixed, yf))

    Y = jnp.concatenate((yu_certain, yu_fixed, yf))
    print("Y:", Y)


    number_Y = Y.shape[0]
    number_Y_certain = Y_certain.shape[0]

    sigma_init = jnp.std(Y)
    sigma_init_yu = jnp.std(Yu)
    sigma_init_yf = jnp.std(yf)
    print(f"sigma_init_yu: {sigma_init_yu}", f"sigma_init_yf: {sigma_init_yf}", f"sigma_init: {sigma_init}",
          sep='\t')

    distances_init = jnp.sqrt((Xu_all_with_noise[:, None, :] - Xu_all_with_noise[None, :, :]) ** 2)
    lengthscale_init = jnp.mean(distances_init, axis=(0, 1))

    kernel_params_only_u = initialize_params_2d(sigma_init_yu, lengthscale_init)
    lengthscale_x = kernel_params_only_u[0][1][0].item()
    lengthscale_t = kernel_params_only_u[0][1][1].item()
    k_ff = compute_kff_rd(Xf, Xf, kernel_params_only_u, lengthscale_x, lengthscale_t)
    k_ff_inv_yf: jnp.ndarray = jnp.linalg.solve(k_ff, yf)
    yf_u = compute_kuf_rd(Xf, Xf, kernel_params_only_u, lengthscale_x, lengthscale_t) @ k_ff_inv_yf

    new_Y = jnp.concatenate((yu_certain, yf_u))
    new_sigma_init = jnp.std(new_Y)
    new_sigma_init_yf = jnp.std(yf_u)
    print(f"new_sigma_init_yu: {sigma_init_yu}", f"new_sigma_init_yf: {new_sigma_init_yf}",
          f"new_sigma_init: {new_sigma_init}", sep='\t')

    # init = initialize_params_2d(sigma_init_yu, lengthscale_init)

    # init = (((jnp.array([0.5], dtype=jnp.float32),
    #                            jnp.array([0.01, 0.08], dtype=jnp.float32))),)

    # log_sigma_init = jnp.log(0.5)
    # log_lx_init = jnp.log(0.01)
    # log_lt_init = jnp.log(0.08)
    # init = (((jnp.array([log_sigma_init], dtype=jnp.float64),
    #           jnp.array([log_lx_init, log_lt_init], dtype=jnp.float64))),)

    # init = (((jnp.array([0.5], dtype=jnp.float64),
    #           jnp.array([0.05, 0.05], dtype=jnp.float64))),)
    init = (((jnp.array([0.5], dtype=jnp.float32),
                jnp.array([0.01, 0.08], dtype=jnp.float32))),)
    # init = (((jnp.array([0.5], dtype=jnp.float64),
    #           jnp.array([0.1, 0.08], dtype=jnp.float64))),)

    print("init:", init)
    param_iter, optimizer_text, lr_text, epoch_text = train_heat_equation_model_2d_rd(init,
                                                                                   Xu_noise,
                                                                                   Xu_fixed,
                                                                                   Xf,
                                                                                   number_Y,
                                                                                   Y, epochs,
                                                                                   learning_rate,
                                                                                   optimizer_in_use,mcmc_text
                                                                                   )

    # param_iter, optimizer_text, lr_text, epoch_text = train_heat_equation_model_2d_rd(init,
    #                                                                                   Xu_fixed,
    #                                                                                   Xf,
    #                                                                                   number_Y_certain,
    #                                                                                   Y_certain, epochs,
    #                                                                                   learning_rate,
    #                                                                                   optimizer_in_use, mcmc_text
    #                                                                                   )

    print("init params:", init)
    # param_iter = init
#nit params:
    # ((Array([0.64007214], dtype=float64),
    # Array([0.80459709, 0.40410401], dtype=float64)),)
    # param_iter = (((jnp.array([0.60191826], dtype=jnp.float32),
    #              jnp.array([0.36601679, 0.05], dtype=jnp.float32))),)
    print("param_iter:", param_iter)
    # init = (((jnp.array([0.1], dtype=jnp.float32),
    #           jnp.array([0.01, 1.0], dtype=jnp.float32))),)
    # print("init:", init)((Array([0.43762741], dtype=float64), Array([0.16809054, 0.00820585], dtype=float64)),)
    #0.60191826], dtype=float64), Array([0.36601679
    X_plot_prediction = jnp.vstack([x_grid_mesh.ravel(), time_grid_mesh.ravel()]).T
    plot_f_inference_rd(param_iter, Xu_fixed, yu_fixed, Xf, yf, added_text, X_plot_prediction, data, learning)
"""
    # %%
    print('start inference')


    # def generate_prior_samples_norm(rng_key, num_samples, prior_mean, prior_cov):
    #
    #     prior_samples = random.multivariate_normal(rng_key, mean=prior_mean.ravel(), cov=prior_cov,
    #                                                shape=(num_samples,))
    #
    #     min_val_0 = jnp.min(prior_samples[0])
    #     max_val_0 = jnp.max(prior_samples[0])
    #
    #     min_val_1 = jnp.min(prior_samples[1])
    #     max_val_1 = jnp.max(prior_samples[1])
    #
    #     prior_samples.at[0].set(2 * (prior_samples[0] - min_val_0) / (max_val_0 - min_val_0) - 1)
    #     prior_samples.at[1].set((prior_samples[1] - min_val_1) / (max_val_1 - min_val_1))
    #
    #     prior_mean_normalized_0 = 2* (prior_mean[0] - min_val_0) / (max_val_0 - min_val_0)-1
    #     prior_mean_normalized_1 = (prior_mean[1] - min_val_1) / (max_val_1 - min_val_1)
    #     prior_mean_normalized = jnp.array([prior_mean_normalized_0, prior_mean_normalized_1])
    #
    #     return prior_samples, prior_mean_normalized


    def generate_prior_samples(rng_key, num_samples, prior_mean, prior_cov):

        prior_samples = random.multivariate_normal(rng_key, mean=prior_mean.ravel(), cov=prior_cov,
                                                   shape=(num_samples,))
        return prior_samples


    def generate_prior_samples_2(rng_key, num_samples, prior_mean, prior_cov):
        prior_samples = random.multivariate_normal(rng_key, mean=prior_mean.ravel(), cov=prior_cov,
                                                   shape=(num_samples,))
        prior_samples_x = jnp.maximum(jnp.minimum(1, prior_samples[:, 0]), -1)
        prior_samples_t = jnp.maximum(jnp.minimum(1, prior_samples[:, 1]), 0)
        prior_samples = jnp.column_stack((prior_samples_x, prior_samples_t))

        return prior_samples

    def find_kde_peak(data):
        kde = gaussian_kde(data)
        x_vals = jnp.linspace(jnp.min(data), jnp.max(data), 1000)
        kde_vals = kde(x_vals)
        peak_idx = jnp.argmax(kde_vals)
        peak = x_vals[peak_idx]
        return peak


    def find_closest_to_gt(gt, values, labels):
        distances = jnp.abs(jnp.array(values) - gt)
        closest_idx = jnp.argmin(distances)
        closest_label = labels[closest_idx]
        return closest_label, distances


    def save_variables(added_text, **variables):
        root_folder = "."
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        filename = f"{added_text}.pkl"
        file_path = os.path.join(root_folder, filename)

        with open(file_path, 'wb') as f:
            pickle.dump(variables, f)
        print(f"Variables saved to {file_path}")

    prior_rng_key, prior_key = random.split(random.PRNGKey(49))
    prior_cov_flat = jnp.kron(jnp.eye(2) * prior_var, jnp.eye(Xu_noise.shape[0]))
    prior_samples_list = generate_prior_samples(prior_key, num_prior_samples, Xu_noise, prior_cov_flat)
    prior_samples = prior_samples_list.reshape(-1, *Xu_noise.shape)
    print("prior_samples list shape:", prior_samples_list.shape)
    print("prior_samples shape:", prior_samples.shape)

    print(f"assumption_sigma={assumption_sigma}")
    rng_key = jax.random.PRNGKey(422)
    all_chains_samples = []

    for chain_id in range(num_chains):
        rng_key, chain_key = random.split(rng_key)
        chain_samples = single_component_metropolis_hasting_rd(chain_key, max_samples, assumption_sigma, Xu_noise,
                                                            jnp.eye(2) * prior_var, param_iter, Xu_fixed, Xf, Y, k)
        all_chains_samples.append(chain_samples)

    all_chains_samples = jnp.array(all_chains_samples)
    num_samples = Xu_noise.shape[0]
    z_uncertain_means = []

    posterior_samples_list = jnp.concatenate(all_chains_samples, axis=0)
    posterior_samples_list = posterior_samples_list.reshape(-1, *Xu_noise.shape)

    print("posterior_samples_list shape:", posterior_samples_list.shape)
    print("posterior_samples_list:", posterior_samples_list)
    print("Xu_certain:", Xu_certain)
    print("Xu_noise:", Xu_noise)
    current_time = datetime.datetime.now().strftime("%M%S")
    added_text = f"REACT_f{number_f}_chains{num_chains}_k{k}_assumption{assumption_sigma}_prior_std{prior_std}_noisestd{noise_std}_init{number_init}_b{number_bound}_{prior_var}_k{k}_{max_samples}_learn{learning}_{current_time}"


    Xu_pred_mean = jnp.mean(posterior_samples_list, axis=0)
    plot_u_pred_rd(Xu_certain, Xf, Xu_noise, noise_std, Xu_pred_mean, prior_var,assumption_sigma,k,max_samples,learning,num_chains,number_f,added_text, X_plot_prediction, data)
    plot_dist_rd(Xu_certain,
                 Xu_noise,
                 Xu_pred_mean,
                 posterior_samples_list,
                 prior_samples_list, number_u, added_text, prior_samples)
    plot_with_noise_rd(number_u, 0, posterior_samples_list, prior_samples_list, Xu_certain, Xu_noise, bw,added_text)
    # plot_and_save_kde_histograms(posterior_samples_list, prior_samples, Xu_certain, Xu_noise, number_u, number_f,
    #                              num_chains, k, assumption_sigma, prior_std, noise_std, number_init, number_bound,
    #                              prior_var, max_samples, added_text)

    save_variables(added_text, Xu_all_without_noise=Xu_all_without_noise, Xu_certain=Xu_certain, Xf=Xf, Xu_noise=Xu_noise,
                   noise_std=noise_std, Xu_pred=Xu_pred_mean, prior_var=prior_var, assumption_sigma=assumption_sigma,
                   k=k, max_samples=max_samples, learning=learning, num_chains=num_chains, number_f=number_f,
                   posterior_samples_list=posterior_samples_list, prior_samples=prior_samples, Y=Y,
                   param_iter=param_iter, Xu_fixed=Xu_fixed, epochs=epochs,
                   learning_rate=learning_rate,
                   optimizer_in_use=optimizer_in_use, number_u_c_for_f=number_u_c_for_f, prior_std=prior_std,
                   number_init=number_init, number_bound=number_bound, data=data, X_plot_prediction=X_plot_prediction,
                   prior_samples_list=prior_samples_list,mcmc_text=mcmc_text,x_grid_mesh_shape=x_grid_mesh_shape)

"""

# %%