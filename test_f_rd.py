import os
import jax
import datetime
import optax
from include.check_hyperparameters import check_hyperparamters
from include.heat2d import plot_u_f, f_xt, plot_u_f_pred, get_u_test_data_2d_qmc, plot_u_pred, u_xt
from include.mcmc_posterior import *
from include.init import ModelInitializer_2d
from include.plot_dist import plot_dist, plot_with_noise, plot_and_save_kde_histograms
from include.plot_pred import plot_and_save_prediction_results
from include.train import train_heat_equation_model_2d
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from include.plot_pred import plot_and_save_prediction_results, prediction_mean, prediction_variance
from scipy.optimize import minimize
from jax import random
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial
import pickle
import os
import jax.scipy.linalg as la
import pickle
import os
import jax.scipy.linalg as la
import gc
from jax.scipy import linalg
import jax.random as random
import matplotlib.pyplot as plt
from skfdiff import Model, Simulation
from scipy.interpolate import CubicSpline
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import griddata

os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_enable_x64", True)

## modified
# 1: generate data through QMC
# 2: rewrite metropolis hasting with adaptive warmup
# 3: rewrite mcmc in JAX
# 4: solution to of sine over x
# 5: change noise data
# 6: mcmc adapted with two types of noise
# 7: plot the distribution of the posterior
# 8: prediction
# 9: plot the prediction
# 10: solved the memory issue
# 11: rewrite the derivative of the kernel
# 12: save all the results in a pickle file and enable the user to load the results
# 13: reaction_diffusion


# %%
learning_rate_pred = 0.01
epoch_pred= 100

noise_std = 0.1
prior_std = 0.04
prior_var = prior_std**2# prior variance
max_samples = 5000
assumption_sigma = 0.05 # step size
k = 0.6
num_chains = 1

bw=2
num_prior_samples = 400
learning_rate = 4e-2
test_num = 2**4
number_u = 2**2 # xt
number_u_only_x = 2**2
number_init = 2**4
number_bound = 2**4
number_u_c_for_f = 2 ** 3

number_u_c_for_f_real = (number_u_c_for_f)**2
number_init_real = number_init-1
number_bound_real = (number_bound+1)*2
number_f = number_u_c_for_f_real+number_init_real+number_bound_real

init_num = number_init
bnum = number_bound

optimizer_in_use = optax.adam
epochs = 1000
added_text = f'{number_u}&{number_u_c_for_f_real}&{number_f}&{number_init}&{number_bound}&{epochs}&{noise_std}'
learning = f'{learning_rate}&{epochs}'


current_time = datetime.datetime.now().strftime("%m%d")
pred_mesh = 200

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

    model = Model("k * (dxxT) - 5*T**3 + 5*T", "T(x)", parameters="k", boundary_conditions="periodic", backend='numpy')
    x = jnp.linspace(-1, 1, 500)
    T = x * x * jnp.cos(jnp.pi * x)
    initial_fields = model.Fields(x=x, T=T, k=1)

    simulation = Simulation(model, initial_fields, dt=0.002, tmax=1, scheme="theta")

    data = [T]
    for t, fields in simulation:
        data.append(fields.T)

    data = jnp.asarray(data)
    timesteps, spatial_points = data.shape

    time_grid = jnp.linspace(0, 1, timesteps)
    x_grid = x
    x_grid_mesh, time_grid_mesh = jnp.meshgrid(x_grid, time_grid)

    num_initial_samples = number_init
    num_boundary_samples = number_bound
    num_samples = number_u_c_for_f

    xu_init = (jnp.cos(jnp.arange(init_num + 1) * jnp.pi / init_num))
    xu_init = xu_init[1:-1, ]
    xu_init = jnp.expand_dims(xu_init, axis=-1)
    print("init_num prior", init_num)
    init_num = xu_init.shape[0]
    print("init num after", init_num)
    tu_init = jnp.zeros(shape=(init_num, 1))

    Yu_init = RBFInterpolator(jnp.expand_dims(x_grid, axis=1), data[0:1, :].T)(xu_init)
    Yu_init = jnp.squeeze(Yu_init)
    Xu_init = jnp.hstack([xu_init, tu_init])

    tu_bound_low = ((jnp.cos(jnp.arange(bnum + 1) * jnp.pi / bnum)) + 1) / 2
    tu_bound_low = jnp.expand_dims(tu_bound_low, axis=-1)
    tu_bound_high = ((jnp.cos(jnp.arange(bnum + 1) * jnp.pi / bnum)) + 1) / 2
    tu_bound_high = jnp.expand_dims(tu_bound_high, axis=-1)
    print("bnum before", bnum)
    bnum = tu_bound_low.shape[0]
    print("bnum after", bnum)
    xu_bound_low = -jnp.ones(shape=(bnum, 1))
    xu_bound_high = jnp.ones(shape=(bnum, 1))

    Yu_bound_low = RBFInterpolator(jnp.expand_dims(time_grid, axis=1), data[:, 0:1])(tu_bound_low)
    Yu_bound_low = jnp.squeeze(Yu_bound_low)
    Xu_bound_low = jnp.hstack([xu_bound_low, tu_bound_low])

    Yu_bound_high = RBFInterpolator(jnp.expand_dims(time_grid, axis=1), data[:, -1:])(tu_bound_high)
    Yu_bound_high = jnp.squeeze(Yu_bound_high)
    Xu_bound_high = jnp.hstack([xu_bound_high, tu_bound_high])
    print("before num_samples", num_samples)

    xu_all = jnp.cos((((2 * jnp.arange(num_samples)) + 1) / (2 * (num_samples))) * jnp.pi)
    tu_all = (jnp.cos((((2 * jnp.arange(num_samples)) + 1) / (2 * (num_samples))) * jnp.pi) + 1) / 2
    Xu_all = jnp.vstack([xu_all, tu_all]).T
    Xu_mesh_all, Tu_mesh_all = jnp.meshgrid(xu_all, tu_all)
    Xu_inner_all = Xu_mesh_all.ravel()
    Tu_inner_all = Tu_mesh_all.ravel()
    num_sample = Xu_inner_all.shape
    print("num_sample num after", num_sample)
    U_inner_all = jnp.vstack([Xu_inner_all, Tu_inner_all]).T

    key_u_rd = random.PRNGKey(3)
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

    Xu_plot = jnp.vstack([Xu_init, Xu_bound_low, Xu_bound_high, U_inner_all, X_u])  #
    yu_plot = jnp.concatenate([Yu_init, Yu_bound_low, Yu_bound_high, Yu_inner_all, yu])

    Xu_fixed = jnp.vstack([Xu_init, Xu_bound_low, Xu_bound_high, U_inner_all])  #
    yu_fixed = jnp.concatenate([Yu_init, Yu_bound_low, Yu_bound_high, Yu_inner_all])

    plt.scatter(Tu_mesh_all, Xu_mesh_all, c="red")
    plt.scatter(tu_init, xu_init, c="blue")
    plt.scatter(tu_bound_low, xu_bound_low, c="green")
    plt.scatter(tu_bound_high, xu_bound_high, c="green")
    plt.scatter(t_u, x_u, color="purple")
    plt.show()
    print("Xu_plot shape", Xu_plot.shape)
    print("Yu_plot shape", yu_plot.shape)
    plt.savefig('data_with_points.png')

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
    beta = 5
    R_u = beta * (yu_fixed ** 3 - yu_fixed)
    Lu_data = -R_u

    Xf = Xu_fixed
    yf = Lu_data

    print("Xf (chosen points):", Xf)
    print("yf (corresponding Lu values):", yf)

    Y = jnp.concatenate((yu_certain, yu_fixed, yf))
    print("Y:", Y)

    number_Y = Y.shape[0]

    sigma_init = jnp.std(Y)
    sigma_init_yu = jnp.std(Yu)
    sigma_init_yf = jnp.std(yf)
    print(f"sigma_init_yu: {sigma_init_yu}", f"sigma_init_yf: {sigma_init_yf}", f"sigma_init: {sigma_init}",
          sep='\t')

    distances_init = jnp.sqrt((Xu_all_with_noise[:, None, :] - Xu_all_with_noise[None, :, :]) ** 2)
    lengthscale_init = jnp.mean(distances_init, axis=(0, 1))


    def initialize_params_2d(sigma_init, lengthscale_init):
        sigma = jnp.array([sigma_init])
        lengthscale = lengthscale_init

        params = (
            (sigma, lengthscale),
        )

        return params

    kernel_params_only_u = initialize_params_2d(sigma_init, lengthscale_init)
    lengthscale_x = kernel_params_only_u[0][1][0].item()
    lengthscale_t = kernel_params_only_u[0][1][1].item()
    k_ff = compute_kff(Xf, Xf, kernel_params_only_u, lengthscale_x, lengthscale_t)
    k_ff_inv_yf: jnp.ndarray = jnp.linalg.solve(k_ff, yf)
    yf_u = compute_kuf(Xf, Xf, kernel_params_only_u, lengthscale_x, lengthscale_t) @ k_ff_inv_yf

    new_Y = jnp.concatenate((yu_certain, yf_u))
    new_sigma_init = jnp.std(new_Y)
    new_sigma_init_yf = jnp.std(yf_u)
    print(f"new_sigma_init_yu: {sigma_init_yu}", f"new_sigma_init_yf: {new_sigma_init_yf}",
          f"new_sigma_init: {new_sigma_init}", sep='\t')

    init = initialize_params_2d(new_sigma_init, lengthscale_init)

    param_iter, optimizer_text, lr_text, epoch_text = train_heat_equation_model_2d(init,
                                                                                   Xu_noise,
                                                                                   Xu_fixed,
                                                                                   Xf,
                                                                                   number_Y,
                                                                                   Y, epochs,
                                                                                   learning_rate,
                                                                                   optimizer_in_use
                                                                                   )

    print("init params:", init)
    print("param_iter:", param_iter)

    # %%
    # # %%
    print("start prediction")
    x_prediction = jnp.linspace(0, 1, pred_mesh)
    t_prediction = jnp.linspace(0, 1, pred_mesh)

    X_prediction, T_prediction = jnp.meshgrid(x_prediction, t_prediction)

    X_plot_prediction = jnp.vstack([X_prediction.ravel(), T_prediction.ravel()]).T

    y_final_mean_list_posterior = []
    y_final_var_list_posterior = []

    y_final_mean_list_prior = []
    y_final_var_list_prior = []


    def compute_K_no(init, Xcz, Xcg):
        params = init
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        lengthscale_x = params[0][1][0].item()
        lengthscale_t = params[0][1][1].item()
        # zz_uu = compute_kuu(Xuz, Xuz, params_kuu)
        # zz_uc = compute_kuu(Xuz, Xcz, params_kuu)
        # zg_uc = compute_kuf(Xuz, Xcg, params, lengthscale_x, lengthscale_t)
        # zz_cu = compute_kuu(Xcz, Xuz, params_kuu)
        zz_cc = compute_kuu(Xcz, Xcz, params_kuu)
        zg_cc = compute_kuf(Xcz, Xcg, params, lengthscale_x, lengthscale_t)
        # gz_cu = compute_kfu(Xcg, Xuz, params, lengthscale_x, lengthscale_t)
        gz_cc = compute_kfu(Xcg, Xcz, params, lengthscale_x, lengthscale_t)
        gg_cc = compute_kff(Xcg, Xcg, params, lengthscale_x, lengthscale_t)
        K = jnp.block([[zz_cc, zg_cc], [gz_cc, gg_cc]])
        return K


    def is_symmetric(matrix, tol=1e-8):
        return jnp.allclose(matrix, matrix.T, atol=tol)


    def compute_condition_number(matrix):
        singular_values = jnp.linalg.svd(matrix, compute_uv=False)
        cond_number = singular_values.max() / singular_values.min()
        return cond_number


    def is_positive_definite(matrix):
        try:
            jnp.linalg.cholesky(matrix)
            return True
        except jnp.linalg.LinAlgError:
            return False


    def add_jitter(matrix, jitter=1e-6):
        jitter_matrix = matrix + jitter * jnp.eye(matrix.shape[0])
        return jitter_matrix





    def gp_predict_diagonal_batch_no(init, Xcz, Xcg, y, x_star, batch_size=2000):
        print("Starting gp_predict_diagonal_batch function")
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        params = init
        K = compute_K_no(init, Xcz, Xcg)
        print("Computed K matrix")
        print("K", K)

        jitter_values = [1e-8, 1e-6, 1e-4, 1e-2]
        for jitter in jitter_values:
            K_jittered = add_jitter(K, jitter)
            pos_def = is_positive_definite(K_jittered)
            cond_number = compute_condition_number(K_jittered)
            print(f"Jitter: {jitter} | Positive Definite: {pos_def} | Condition Number: {cond_number}")
            if pos_def and cond_number < 1e6:
                break

        mu_star = []
        sigma_star_diag = []
        K_jittered = add_jitter(K, jitter)
        try:
            K_inv_y = linalg.solve(K_jittered, y, assume_a='pos')
            print("Solved K_inv_y successfully.")
        except Exception as e:
            print(f"Error in solving linear system: {e}")

        if jnp.isnan(K_inv_y).any() or jnp.isinf(K_inv_y).any():
            print("Result contains NaN or Inf values.")
        else:
            print("Result is valid.")
        symmetric = is_symmetric(K)
        print(f"Is K symmetric? {symmetric}")
        cond_number = compute_condition_number(K)
        print(f"Condition number of K: {cond_number}")
        # K_inv_y = la.solve(K, y, assume_a='pos')
        # print("K_inv_y ", K_inv_y )
        pos_def = is_positive_definite(K)
        print(f"Is K positive definite? {pos_def}")

        for i in range(0, x_star.shape[0], batch_size):
            x_star_batch = x_star[i:i + batch_size]

            # k_zz_u_star = compute_kuu(z_prior, x_star_batch, params_kuu)
            k_zz_c_star = compute_kuu(Xcz, x_star_batch, params_kuu)
            k_gz_c_star = compute_kfu(Xcg, x_star_batch, params, params[0][1][0].item(), params[0][1][1].item())

            k_x_star_batch = jnp.vstack((k_zz_c_star, k_gz_c_star))
            mu_star_batch = jnp.dot(k_x_star_batch.T, K_inv_y)

            K_inv_k_x_star_batch = la.solve(K, k_x_star_batch, assume_a='pos')
            sigma_star_batch = compute_kuu(x_star_batch, x_star_batch, params_kuu) - jnp.dot(k_x_star_batch.T,
                                                                                             K_inv_k_x_star_batch)
            sigma_star_batch_diag = sigma_star_batch.diagonal()

            mu_star.append(mu_star_batch)
            sigma_star_diag.append(sigma_star_batch_diag)

        mu_star = jnp.concatenate(mu_star, axis=0)
        sigma_star_diag = jnp.concatenate(sigma_star_diag, axis=0).flatten()

        del K_inv_y, K, k_zz_c_star, k_gz_c_star, k_x_star_batch, K_inv_k_x_star_batch
        gc.collect()
        return mu_star.flatten(), sigma_star_diag


    Y_no = jnp.concatenate((yu_fixed, yf))
    y_final_mean, y_final_var = gp_predict_diagonal_batch_no(param_iter, Xu_fixed, Xf, Y_no, X_plot_prediction)
    print("Prediction mean shape: ", y_final_mean.shape)
    print("Prediction variance shape: ", y_final_var.shape)

    y_final_mean_list_posterior.append(y_final_mean.T)
    y_final_var_list_posterior.append(y_final_var.T)

    gc.collect()
    jax.clear_caches()


    def save_variables(added_text, **variables):
        root_folder = "."
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        filename = f"Pred_{added_text}.pkl"
        file_path = os.path.join(root_folder, filename)

        with open(file_path, 'wb') as f:
            pickle.dump(variables, f)
        print(f"Variables saved to {file_path}")


    y_final_mean_list_posterior = jnp.array(y_final_mean_list_posterior)
    y_final_var_list_posterior = jnp.array(y_final_var_list_posterior)

    print("posterior Prediction mean shape: ", y_final_mean_list_posterior.shape)
    print("posterior Prediction variance shape: ", y_final_var_list_posterior.shape)

    # y_final_mean_posterior = prediction_mean(y_final_mean_list_posterior)

    # y_final_var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior)

    print("-------------------end prediction-------------------")

    u_values_gt = u_xt(X_plot_prediction)

    gp_mean_posterior = y_final_mean_list_posterior.reshape(pred_mesh, pred_mesh)
    u_values_gt = u_values_gt.reshape(pred_mesh, pred_mesh)


    # abs_diff_gt_gp = jnp.abs(u_values_gt - gp_mean_posterior)
    #
    # var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior).reshape(pred_mesh,
    #                                                                                                      pred_mesh)
    #

    def plot_combine(Xf, u_values_gt,
                     gp_mean_posterior,
                     added_text):

        plot_titles = [
            'Ground Truth',
            'GP Mean Prediction (Posterior)',
        ]
        cmap1 = 'GnBu'
        plot_data = [
            (u_values_gt, cmap1),
            (gp_mean_posterior, cmap1),

        ]

        row1_min = min(jnp.min(plot_data[0][0]), jnp.min(plot_data[1][0]))
        row1_max = max(jnp.max(plot_data[0][0]), jnp.max(plot_data[1][0]))

        # row2_min = min(jnp.min(plot_data[3][0]), jnp.min(plot_data[4][0]), jnp.min(plot_data[5][0]))
        # row2_max = max(jnp.max(plot_data[3][0]), jnp.max(plot_data[4][0]), jnp.max(plot_data[5][0]))

        fig, axs = plt.subplots(1, 2, figsize=(18, 12))

        for i in range(2):
            data, cmap = plot_data[i]
            im = axs[i].imshow(data, cmap=cmap, vmin=row1_min, vmax=row1_max, extent=[0, 1, 0, 1])
            axs[i].set_title(plot_titles[i])
            axs[i].scatter(Xf[:, 0], Xf[:, 1], color="red")
            fig.colorbar(im, ax=axs[i])

        # for i in range(3):
        #     data, cmap = plot_data[i + 3]
        #     im = axs[1, i].imshow(data, cmap=cmap, vmin=row2_min, vmax=row2_max)
        #     axs[1, i].set_title(plot_titles[i + 3])
        #     fig.colorbar(im, ax=axs[1, i])

        plt.tight_layout()
        current_time = datetime.datetime.now().strftime("%M%S")
        plt.savefig(f"combined_plot_{added_text}_{current_time}.png")
        plt.show()


    plot_combine(Xf, u_values_gt,
                 gp_mean_posterior,
                 added_text)

    print(gp_mean_posterior.shape)
    print(u_values_gt.shape)

