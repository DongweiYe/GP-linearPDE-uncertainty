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
max_samples = 2000
assumption_sigma = 0.004 # step size
k = 0.6
num_chains = 1

bw=2
num_prior_samples = 400
learning_rate = 4e-2
test_num = 2**4
number_u = 2**2 # xt
number_u_only_x = 2**2
number_f = 2**3
number_init = 2**4
number_bound = 2**4

number_f_real = (number_f)**2
number_init_real = number_init-1
number_bound_real = number_bound+1

param_text = "para"
optimizer_in_use = optax.adam
sample_num = 10
epochs = 1000
added_text = f'number_f_real_{number_f_real}_{number_u}&{number_u_only_x}&fnumber{number_f}&{number_init}&{number_bound}&{sample_num}&{epochs}&{noise_std}'
weight_decay = 1e-5
DEEP_FLAG = False
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
    print("param_text:", param_text, "\n")
    print("optimizer_in_use:", optimizer_in_use, "\n")
    print("sample_num:", sample_num, "\n")
    print("epochs:", epochs, "\n")
    print("added_text:", added_text, "\n")
    print("learning_rate:", learning_rate, "\n")
    print("weight_decay:", weight_decay, "\n")
    print("DEEP_FLAG:", DEEP_FLAG, "\n")

    model_initializer = ModelInitializer_2d(number_u=number_u, number_f=number_f, sample_num=sample_num,
                                            number_init=number_init, number_bound=number_bound, noise_std=noise_std,
                                            number_u_only_x=number_u_only_x)
    Xu_certain = model_initializer.Xu_certain
    Xu_noise = model_initializer.Xu_noise
    yu_certain = model_initializer.yu_certain
    print("Xu_certain:", Xu_certain)
    print("Xu_noise:", Xu_noise)
    print("yu_certain:", yu_certain)

    Xu_fixed = model_initializer.Xu_fixed
    Yu_fixed = model_initializer.Yu_fixed
    print("Xu_fixed:", Xu_fixed)
    print("Yu_fixed:", Yu_fixed)

    Xu_all_noise = model_initializer.Xu_with_noise
    Xu_without_noise = model_initializer.Xu_without_noise
    Yu = model_initializer.Yu
    print("Xu_all_noise:", Xu_all_noise)
    print("Yu:", Yu)

    Xf = model_initializer.Xf
    yf = model_initializer.yf
    print("Xf:", Xf)
    print("yf:", yf)

    def plot_points_f(Xf):
        plt.scatter(Xf[:, 0], Xf[:, 1], c="blue")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.title("Xf")
        plt.savefig("Xf.png")

    plot_points_f(Xf)

    Y = model_initializer.Y
    X_with_noise = model_initializer.X_with_noise

    X_without_noise = model_initializer.X_without_noise
    print("Y:", Y)
    print("X:", X_with_noise)

    xtest = model_initializer.xtest
    ytest = model_initializer.ytest

    # number_u = model_initializer.number_u
    # number_f = model_initializer.number_f
    # number_init = model_initializer.number_init
    # number_bound = model_initializer.number_bound
    number_Y = Y.shape[0]

    init = model_initializer.heat_params_init
    print("Xu_certain:", Xu_certain)
    print("Xu_noise:", Xu_noise)
    # plot_u_f(Xu_without_noise, Xf, Xu_noise, noise_std)

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


    Y_no = jnp.concatenate((Yu_fixed, yf))
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
