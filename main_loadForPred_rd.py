
import jax
import datetime
from include.heat2d import compute_kuu_rd, compute_kuf_rd, compute_kfu_rd, compute_kff_rd
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from jax import random
from scipy.stats import gaussian_kde
import pickle
import os
import jax.scipy.linalg as la
import gc
import jaxlib
from include.mcmc_posterior import compute_K, compute_K_rd
from include.plot_dist import plot_dist, plot_with_noise, plot_and_save_kde_histograms, plot_dist_rd, plot_with_noise_rd
from include.plot_pred import plot_and_save_prediction_results, prediction_mean, prediction_variance, \
    plot_and_save_prediction_results_combine, plot_and_save_prediction_results_combine_rd, \
    plot_and_save_prediction_results_rd
from include.plot_pred_test import plot_prediction_results_test
from include.train import train_heat_equation_model_2d, train_heat_equation_model_2d_rd

os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_enable_x64", True)

# bw = 1
# num_prior_samples = 200
current_time = datetime.datetime.now().strftime("%m%d")
# learning_rate_pred = 0.004
# epoch_pred = 500
# pred_mesh = 200

text = "REACT_f458_chains1_k0.6_assumption0.01_prior_std0.06_noisestd0.04_init32_b32_0.0036_k0.6_1000_learnlr0.1&800_4355.pkl"
load_path = f"results/datas/trained_params/1213"


# %%s

if __name__ == '__main__':
    # %%
    print('start inference')


    # def generate_prior_samples(rng_key, num_samples, prior_mean, prior_cov):
    #     prior_samples = random.multivariate_normal(rng_key, mean=prior_mean.ravel(), cov=prior_cov,
    #                                                shape=(num_samples,))
    #
    #     # min_val = jnp.min(prior_samples)
    #     # max_val = jnp.max(prior_samples)
    #     #
    #     # prior_samples = (prior_samples - min_val) / (max_val - min_val)
    #
    #     return prior_samples


    # def find_kde_peak(data):
    #     kde = gaussian_kde(data)
    #     x_vals = jnp.linspace(jnp.min(data), jnp.max(data), 1000)
    #     kde_vals = kde(x_vals)
    #     peak_idx = jnp.argmax(kde_vals)
    #     peak = x_vals[peak_idx]
    #     return peak

    #
    # def find_closest_to_gt(gt, values, labels):
    #     distances = jnp.abs(jnp.array(values) - gt)
    #     closest_idx = jnp.argmin(distances)
    #     closest_label = labels[closest_idx]
    #     return closest_label, distances

    def load_variables(text, load_path):
        print(f"Loading data from {load_path}")
        filename = f"{text}"

        file_path = os.path.join(load_path, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No data found at {file_path}")
        with open(file_path, 'rb') as f:
            load_variables = pickle.load(f)
        print(f"Variables loaded from {file_path}")
        return load_variables


    variables = load_variables(text, load_path)
    #
    # Xu_without_noise = variables['Xu_without_noise']
    Xu_certain = variables['Xu_certain']
    Xf = variables['Xf']
    Xu_noise = variables['Xu_noise']
    noise_std = variables['noise_std']
    Xu_pred = variables['Xu_pred']
    prior_var = variables['prior_var']
    assumption_sigma = variables['assumption_sigma']
    k = variables['k']
    max_samples = variables['max_samples']
    learning = variables['learning']
    num_chains = variables['num_chains']
    number_f = variables['number_f']
    posterior_samples_list = variables['posterior_samples_list']
    prior_samples = variables['prior_samples']
    Y = variables['Y']
    number_u = Xu_noise.shape[0]
    param_iter = variables['param_iter']
    Xu_fixed = variables['Xu_fixed']
    epochs = variables['epochs']
    learning_rate = variables['learning_rate']
    optimizer_in_use = variables['optimizer_in_use']
    number_u_c_for_f = variables['number_u_c_for_f']
    prior_std = variables['prior_std']
    number_init = variables['number_init']
    number_bound = variables['number_bound']
    data = variables['data']
    X_plot_prediction = variables['X_plot_prediction']
    prior_samples_list = variables['prior_samples_list']
    mcmc_text = variables['mcmc_text']
    x_grid_mesh_shape = variables['x_grid_mesh_shape']
    added_text = f"rd_Predction_f{number_f}_chains{num_chains}_k{k}_assumption{assumption_sigma}_noisestd{noise_std}_{prior_var}_k{k}_{max_samples}_{current_time}"

    Xu_pred_mean = jnp.mean(posterior_samples_list, axis=0)

    print("Xu_noise:", Xu_noise)
    print("number_u:", number_u)
    # plot_u_pred_rd(Xu_certain, Xf, Xu_noise, noise_std, Xu_pred_mean, prior_var,assumption_sigma,k,max_samples,learning,num_chains,number_f,added_text, X_plot_prediction, data)
    # plot_dist_rd(Xu_certain,
    #              Xu_noise,
    #              Xu_pred_mean,
    #              posterior_samples_list,
    #              prior_samples_list, number_u, added_text)
    # plot_with_noise_rd(number_u, 0, posterior_samples_list, prior_samples_list, Xu_certain, Xu_noise, bw,added_text)

    print('end inference')

# %%
# # %%
    print("start prediction")
    # x_prediction = jnp.linspace(0, 1, pred_mesh)
    # t_prediction = jnp.linspace(0, 1, pred_mesh)
    #
    # X_prediction, T_prediction = jnp.meshgrid(x_prediction, t_prediction)
    #
    # X_plot_prediction = jnp.vstack([X_prediction.ravel(), T_prediction.ravel()]).T

    # y_final_mean_list_posterior = jnp.empty((0, X_plot_prediction.shape[0]))
    # y_final_var_list_posterior = jnp.empty((0, X_plot_prediction.shape[0]))
    #
    # y_final_mean_list_prior = jnp.empty((0, X_plot_prediction.shape[0]))
    # y_final_var_list_prior = jnp.empty((0, X_plot_prediction.shape[0]))

    pred_mesh = X_plot_prediction.shape[0]
    y_final_mean_list_posterior = []
    y_final_var_list_posterior = []

    y_final_mean_list_prior = []
    y_final_var_list_prior = []


    # def compute_joint_K(init, z_prior, Xcz, Xcg, x_star):
    #     Xuz = z_prior
    #     params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    #     params = init
    #     K = compute_K(init, z_prior, Xcz, Xcg)
    #     lengthscale_x = params[0][1][0].item()
    #     lengthscale_t = params[0][1][1].item()
    #
    #     k_zz_u_star = compute_kuu_rd(Xuz, x_star, params_kuu)
    #     k_zz_c_star = compute_kuu_rd(Xcz, x_star, params_kuu)
    #     k_gz_c_star = compute_kfu_rd(Xcg, x_star, params, lengthscale_x, lengthscale_t)
    #
    #     k_x_star = jnp.vstack((k_zz_u_star, k_zz_c_star, k_gz_c_star))
    #
    #     k_x_star_x_star = compute_kuu_rd(x_star, x_star, params_kuu)
    #
    #     joint_K = jnp.block([[K, k_x_star], [k_x_star.T, k_x_star_x_star]])
    #
    #     return joint_K


    # def gp_predict(init, z_prior, Xcz, Xcg, y, x_star):
    #     params = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    #
    #     K = compute_K(init, z_prior, Xcz, Xcg)
    #
    #     k_zz_u_star = heat_equation_kuu(z_prior, x_star, params)
    #     k_zz_c_star = heat_equation_kuu(Xcz, x_star, params)
    #     k_gz_c_star = heat_equation_kfu(Xcg, x_star, params)
    #
    #     k_x_star = jnp.vstack((k_zz_u_star, k_zz_c_star, k_gz_c_star))
    #
    #     k_x_star_x_star = heat_equation_kuu(x_star, x_star, params)
    #
    #     K_inv_y = la.solve(K + 1e-6 * jnp.eye(K.shape[0]), y.reshape(-1, 1), assume_a='pos')
    #     K_inv_k_x_star = la.solve(K + 1e-6 * jnp.eye(K.shape[0]), k_x_star, assume_a='pos')
    #
    #     mu_star = jnp.dot(k_x_star.T, K_inv_y)
    #
    #     sigma_star = k_x_star_x_star - jnp.einsum('ij,ij->i', k_x_star.T, K_inv_k_x_star.T).reshape(-1, 1)
    #
    #     return mu_star.flatten(), sigma_star


    # def blockwise_matrix_multiply(A, B, block_size):
    #     M, N = A.shape
    #     _, P = B.shape
    #     C = jnp.zeros((M, P))
    #     for i in range(0, M, block_size):
    #         for j in range(0, P, block_size):
    #             C = C.at[i:i + block_size, j:j + block_size].add(
    #                 jnp.dot(A[i:i + block_size, :], B[:, j:j + block_size]))
    #     return C

    # def gp_predict(init, z_prior, Xcz, Xcg, y, x_star):
    #     print("Starting gp_predict function")
    #
    #     params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    #     params = init
    #     K = compute_K(init, z_prior, Xcz, Xcg)
    #     print("Computed K matrix")
    #
    #     lengthscale_x = params[0][1][0].item()
    #     lengthscale_t = params[0][1][1].item()
    #
    #     k_zz_u_star = compute_kuu_rd(z_prior, x_star, params_kuu)
    #     k_zz_c_star = compute_kuu_rd(Xcz, x_star, params_kuu)
    #     k_gz_c_star = compute_kfu_rd(Xcg, x_star, params, lengthscale_x, lengthscale_t)
    #
    #     k_x_star = jnp.vstack((k_zz_u_star, k_zz_c_star, k_gz_c_star))
    #     k_x_star_x_star = compute_kuu_rd(x_star, x_star, params_kuu)
    #     del k_zz_u_star, k_zz_c_star, k_gz_c_star, params_kuu, params
    #
    #     K_inv_y = la.solve(K, y, assume_a='pos')
    #     K_inv_k_x_star = la.solve(K, k_x_star, assume_a='pos')
    #     mu_star_gpu = jnp.dot(k_x_star.T, K_inv_y)
    #     del K, K_inv_y
    #     k_x_star_T = k_x_star.T
    #     del k_x_star
    #     k_x_star_T_K_inv_k_x_star = k_x_star_T@K_inv_k_x_star
    #     del k_x_star_T, K_inv_k_x_star
    #     sigma_star_gpu = k_x_star_x_star - k_x_star_T_K_inv_k_x_star
    #
    #     del k_x_star_x_star
    #     gc.collect()
    #     return mu_star_gpu.flatten(), sigma_star_gpu

    # def gp_predict_diagonal(init, z_prior, Xcz, Xcg, y, x_star):
    #     print("Starting gp_predict_diagonal function")
    #     params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    #     params = init
    #     K = compute_K(init, z_prior, Xcz, Xcg)
    #     print("Computed K matrix")
    #
    #     K_inv_y = la.solve(K, y, assume_a='pos')
    #
    #     mu_star = []
    #     sigma_star_diag = []
    #
    #     for i in range(x_star.shape[0]):
    #         x_star_i = x_star[i:i + 1]
    #
    #         k_zz_u_star = compute_kuu_rd(z_prior, x_star_i, params_kuu)
    #         k_zz_c_star = compute_kuu_rd(Xcz, x_star_i, params_kuu)
    #         k_gz_c_star = compute_kfu_rd(Xcg, x_star_i, params, params[0][1][0].item(), params[0][1][1].item())
    #
    #         k_x_star_i = jnp.vstack((k_zz_u_star, k_zz_c_star, k_gz_c_star))
    #         mu_star_i = jnp.dot(k_x_star_i.T, K_inv_y)
    #
    #         K_inv_k_x_star_i = la.solve(K, k_x_star_i, assume_a='pos')
    #         sigma_star_i = compute_kuu_rd(x_star_i, x_star_i, params_kuu) - jnp.dot(k_x_star_i.T, K_inv_k_x_star_i)
    #
    #         mu_star.append(mu_star_i)
    #         sigma_star_diag.append(sigma_star_i)
    #
    #     mu_star = jnp.concatenate(mu_star, axis=0)
    #     sigma_star_diag = jnp.concatenate(sigma_star_diag, axis=0).flatten()
    #
    #     del K_inv_y, K, mu_star_i, sigma_star_i, k_zz_u_star, k_zz_c_star, k_gz_c_star, k_x_star_i, K_inv_k_x_star_i
    #     gc.collect()
    #     return mu_star.flatten(), sigma_star_diag
    #     # # CPU
    #     # K_cpu = jax.device_put(K + 1e-6 * jnp.eye(K.shape[0]), device=jax.devices("cpu")[0])
    #     # y_cpu = jax.device_put(y.reshape(-1, 1), device=jax.devices("cpu")[0])
    #     # k_x_star_cpu = jax.device_put(k_x_star, device=jax.devices("cpu")[0])
    #     # print("Moved data to CPU")
    #     # K_inv_y = la.solve(K_cpu, y_cpu, assume_a='pos')
    #     # K_inv_k_x_star = la.solve(K_cpu, k_x_star_cpu, assume_a='pos')
    #     # mu_star_cpu = jnp.dot(k_x_star_cpu.T, K_inv_y)
    #     # sigma_star_cpu = k_x_star_x_star - jnp.einsum('ij,ij->i', k_x_star_cpu.T, K_inv_k_x_star.T).reshape(-1, 1)
    #     # ## GPU
    #     # mu_star_gpu = jax.device_put(mu_star_cpu, device=jax.devices("gpu")[0])
    #     # sigma_star_gpu = jax.device_put(sigma_star_cpu, device=jax.devices("gpu")[0])
    #     # print("Moved results back to GPU")
    #     # # try:
    #     # #     sigma_star_gpu = jax.device_put(sigma_star_cpu, device=jax.devices("gpu")[0])
    #     # # except jaxlib.xla_extension.XlaRuntimeError:
    #     # #     print("GPU lack of memory")
    #     # #     raise
    #     # del K_cpu, y_cpu, k_x_star_cpu, K_inv_y, K_inv_k_x_star, mu_star_cpu, sigma_star_cpu
    #     #
    #     ## block_size = 100
    #     # block_result = blockwise_matrix_multiply(k_x_star_T, K_inv_k_x_star, block_size=block_size)
    #     # del k_x_star_T, K_inv_k_x_star
    #     # sigma_star_gpu = k_x_star_x_star - block_result
    #     # print("block_size=", block_size)


    def compute_K_no(init, Xcz, Xcg):
        params = init
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        lengthscale_x = params[0][1][0].item()
        lengthscale_t = params[0][1][1].item()
        # zz_uu = compute_kuu(Xuz, Xuz, params_kuu)
        # zz_uc = compute_kuu(Xuz, Xcz, params_kuu)
        # zg_uc = compute_kuf(Xuz, Xcg, params, lengthscale_x, lengthscale_t)
        # zz_cu = compute_kuu(Xcz, Xuz, params_kuu)
        zz_cc = compute_kuu_rd(Xcz, Xcz, params_kuu)
        zg_cc = compute_kuf_rd(Xcz, Xcg, params, lengthscale_x, lengthscale_t)
        # gz_cu = compute_kfu(Xcg, Xuz, params, lengthscale_x, lengthscale_t)
        gz_cc = compute_kfu_rd(Xcg, Xcz, params, lengthscale_x, lengthscale_t)
        gg_cc = compute_kff_rd(Xcg, Xcg, params, lengthscale_x, lengthscale_t)
        K = jnp.block([[zz_cc, zg_cc], [gz_cc, gg_cc]])
        print("Computed K matrix shape compute K _ no ", K.shape)
        return K

    def gp_predict_diagonal_batch(init, z_prior, Xcz, Xcg, y, x_star, batch_size=2000):
        print("Starting gp_predict_diagonal_batch function")
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        params = init
        K = compute_K_rd(init, z_prior, Xcz, Xcg)
        print("Computed K matrix")

        K_inv_y = la.solve(K, y, assume_a='pos')

        mu_star = []
        sigma_star_diag = []

        for i in range(0, x_star.shape[0], batch_size):
            x_star_batch = x_star[i:i + batch_size]

            k_zz_u_star = compute_kuu_rd(z_prior, x_star_batch, params_kuu)
            k_zz_c_star = compute_kuu_rd(Xcz, x_star_batch, params_kuu)
            k_gz_c_star = compute_kfu_rd(Xcg, x_star_batch, params, params[0][1][0].item(), params[0][1][1].item())
            k_x_star_batch = jnp.vstack((k_zz_u_star, k_zz_c_star, k_gz_c_star))
            mu_star_batch = jnp.dot(k_x_star_batch.T, K_inv_y)
            # k_x_star_batch = jnp.vstack((k_zz_c_star, k_gz_c_star))
            # mu_star_batch = jnp.dot(k_x_star_batch.T, K_inv_y)

            K_inv_k_x_star_batch = la.solve(K, k_x_star_batch, assume_a='pos')
            sigma_star_batch = compute_kuu_rd(x_star_batch, x_star_batch, params_kuu) - jnp.dot(k_x_star_batch.T,
                                                                                             K_inv_k_x_star_batch)
            sigma_star_batch_diag = sigma_star_batch.diagonal()

            mu_star.append(mu_star_batch)
            sigma_star_diag.append(sigma_star_batch_diag)
            print("k_x_star_batch shape:", k_x_star_batch.shape)
            print("K_inv_y shape:", K_inv_y.shape)
            print("mu_star_batch shape:", mu_star_batch.shape)
            print("sigma_star_batch_diag shape:", sigma_star_batch_diag.shape)

        mu_star = jnp.concatenate(mu_star, axis=0)
        sigma_star_diag = jnp.concatenate(sigma_star_diag, axis=0).flatten()

        del K_inv_y, K, k_zz_c_star, k_gz_c_star, k_x_star_batch, K_inv_k_x_star_batch
        gc.collect()
        return mu_star.flatten(), sigma_star_diag


    def gp_predict_diagonal_batch2(init, z_prior, Xcz, Xcg, y, x_star, batch_size=2000):
        print("Starting gp_predict_diagonal_batch function")
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        params = init
        # K = compute_K(init, z_prior, Xcz, Xcg)
        Xuc = jnp.concatenate((z_prior, Xcz))
        K = compute_K_no(init, Xuc, Xcg)
        print("Computed K matrix")

        K_inv_y = la.solve(K, y, assume_a='pos')

        mu_star = []
        sigma_star_diag = []

        for i in range(0, x_star.shape[0], batch_size):
            x_star_batch = x_star[i:i + batch_size]

            Xz = jnp.concatenate((z_prior, Xcz))
            k_zz_c_star = compute_kuu_rd(Xz, x_star_batch, params_kuu)
            k_gz_c_star = compute_kfu_rd(Xcg, x_star_batch, params, params[0][1][0].item(), params[0][1][1].item())
            # k_zz_u_star = compute_kuu(z_prior, x_star_batch, params_kuu)
            # k_zz_c_star = compute_kuu(Xcz, x_star_batch, params_kuu)
            # k_gz_c_star = compute_kfu(Xcg, x_star_batch, params, params[0][1][0].item(), params[0][1][1].item())
            # k_x_star_batch = jnp.vstack((k_zz_u_star, k_zz_c_star, k_gz_c_star))
            # mu_star_batch = jnp.dot(k_x_star_batch.T, K_inv_y)
            k_x_star_batch = jnp.vstack((k_zz_c_star, k_gz_c_star))
            mu_star_batch = jnp.dot(k_x_star_batch.T, K_inv_y)

            K_inv_k_x_star_batch = la.solve(K, k_x_star_batch, assume_a='pos')
            sigma_star_batch = compute_kuu_rd(x_star_batch, x_star_batch, params_kuu) - jnp.dot(k_x_star_batch.T,
                                                                                             K_inv_k_x_star_batch)
            sigma_star_batch_diag = sigma_star_batch.diagonal()

            mu_star.append(mu_star_batch)
            sigma_star_diag.append(sigma_star_batch_diag)

        mu_star = jnp.concatenate(mu_star, axis=0)
        sigma_star_diag = jnp.concatenate(sigma_star_diag, axis=0).flatten()

        del K_inv_y, K, k_zz_c_star, k_gz_c_star, k_x_star_batch, K_inv_k_x_star_batch
        gc.collect()
        return mu_star.flatten(), sigma_star_diag



    for i in range(posterior_samples_list.shape[0]):
        Xu_sample = posterior_samples_list[i, :, :]
        mcmc_text=f"mcmc"
        # param_iter, _, _, _ = train_heat_equation_model_2d(param_iter,
        #                                                    Xu_sample,
        #                                                    Xu_fixed,
        #                                                    Xf,
        #                                                    Y.shape[0],
        #                                                    Y, epoch_pred,
        #                                                    learning_rate_pred,
        #                                                    optimizer_in_use,mcmc_text)

        lengthscale = param_iter[-1][1]
        sigma = param_iter[-1][0]

        y_final_mean, y_final_var = gp_predict_diagonal_batch(param_iter, Xu_sample, Xu_fixed, Xf, Y, X_plot_prediction)
        print("Prediction mean shape: ", y_final_mean.shape)
        print("Prediction variance shape: ", y_final_var.shape)

        # y_final_mean_list_posterior = jnp.vstack((y_final_mean_list_posterior, y_final_mean.T))
        # y_final_var_list_posterior = jnp.vstack((y_final_var_list_posterior, y_final_var.T))
        y_final_mean_list_posterior.append(y_final_mean.T)
        y_final_var_list_posterior.append(y_final_var.T)

        del Xu_sample, y_final_mean, y_final_var

        gc.collect()
        jax.clear_caches()
        print("posterior memory cleaned up after iteration", i)

    # prior_samples_reshaped = prior_samples.reshape(prior_samples.shape[0], -1, 2)
    prior_samples_reshaped = prior_samples
    for i in range(prior_samples_reshaped.shape[0]):
        Xu_sample_prior = prior_samples_reshaped[i, :, :]

        # param_iter, _, _, _ = train_heat_equation_model_2d_rd(param_iter,
        #                                                    Xu_sample_prior,
        #                                                    Xu_fixed,
        #                                                    Xf,
        #                                                    Y.shape[0],
        #                                                    Y, epoch_pred,
        #                                                    learning_rate_pred,
        #                                                    optimizer_in_use,mcmc_text)

        lengthscale = param_iter[-1][1]
        sigma = param_iter[-1][0]

        y_final_mean_prior, y_final_var_prior = gp_predict_diagonal_batch(param_iter, Xu_sample_prior, Xu_fixed, Xf, Y,
                                                           X_plot_prediction)
        print("prior Prediction mean shape: ", y_final_mean_prior.shape)
        print("prior Prediction variance shape: ", y_final_var_prior.shape)
        # y_final_mean_list_prior = jnp.vstack((y_final_mean_list_prior, y_final_mean_prior.reshape(1, -1)))
        # y_final_var_list_prior = jnp.vstack((y_final_var_list_prior, y_final_var_prior.reshape(1, -1)))
        y_final_mean_list_prior.append(y_final_mean_prior.T)
        y_final_var_list_prior.append(y_final_var_prior.T)

        del Xu_sample_prior, y_final_mean_prior, y_final_var_prior

        gc.collect()
        jax.clear_caches()
        print("prior memory cleaned up after iteration", i)

    def save_variables(added_text, **variables):
        root_folder = "."
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        filename = f"Pred_{added_text}.pkl"
        file_path = os.path.join(root_folder, filename)

        with open(file_path, 'wb') as f:
            pickle.dump(variables, f)
        print(f"Variables saved to {file_path}")


    print("prior_samples_reshaped: ", prior_samples_reshaped)
    print("posterior_samples_list: ", posterior_samples_list)

    #gpu
    # y_final_mean_list_posterior = jnp.array(y_final_mean_list_posterior)
    # y_final_var_list_posterior = jnp.array(y_final_var_list_posterior)
    #
    # print("posterior Prediction mean shape: ", y_final_mean_list_posterior.shape)
    # print("posterior Prediction variance shape: ", y_final_var_list_posterior.shape)
    #
    # y_final_mean_list_prior = jnp.array(y_final_mean_list_prior)
    # y_final_var_list_prior = jnp.array(y_final_var_list_prior)
    #
    # print("prior Prediction mean shape: ", y_final_mean_list_prior.shape)
    # print("prior Prediction variance shape: ", y_final_var_list_prior.shape)

    # CPU
    # y_final_mean_list_posterior = jax.device_put(y_final_mean_list_posterior, device=jax.devices("cpu")[0])
    # y_final_var_list_posterior = jax.device_put(y_final_var_list_posterior, device=jax.devices("cpu")[0])

    y_final_mean_list_posterior = jnp.array(y_final_mean_list_posterior)
    y_final_var_list_posterior = jnp.array(y_final_var_list_posterior)

    print("posterior Prediction mean shape: ", y_final_mean_list_posterior.shape)
    print("posterior Prediction variance shape: ", y_final_var_list_posterior.shape)

    # y_final_mean_list_prior = jax.device_put(y_final_mean_list_prior, device=jax.devices("cpu")[0])
    # y_final_var_list_prior = jax.device_put(y_final_var_list_prior, device=jax.devices("cpu")[0])
    y_final_mean_list_prior = jnp.array(y_final_mean_list_prior)
    y_final_var_list_prior = jnp.array(y_final_var_list_prior)
    print("prior Prediction mean shape: ", y_final_mean_list_prior.shape)
    print("prior Prediction variance shape: ", y_final_var_list_prior.shape)

    y_final_mean_posterior = prediction_mean(y_final_mean_list_posterior)

    #y_final_var_list_posterior_diag = jnp.diagonal(y_final_var_list_posterior, axis1=1, axis2=2)
    y_final_var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior)
    y_final_mean_prior = prediction_mean(y_final_mean_list_prior)

    #y_final_var_list_prior_diag = jnp.diagonal(y_final_var_list_prior, axis1=1, axis2=2)
    y_final_var_prior = prediction_variance(y_final_mean_list_prior, y_final_var_list_prior)
    print("final posterior Prediction mean shape: ", y_final_mean_posterior.shape)
    print("final posterior Prediction variance shape: ", y_final_var_posterior.shape)
    print("final prior Prediction mean shape: ", y_final_mean_prior.shape)
    print("final prior Prediction variance shape: ", y_final_var_prior.shape)
    print("-------------------end prediction-------------------")

    u_values_gt = data
    gp_mean_posterior = prediction_mean(y_final_mean_list_posterior).reshape(x_grid_mesh_shape)
    abs_diff_gt_gp = jnp.abs(u_values_gt - gp_mean_posterior)
    gp_mean_prior = prediction_mean(y_final_mean_list_prior).reshape(x_grid_mesh_shape)
    abs_diff_prior = jnp.abs(u_values_gt - gp_mean_prior)
    var_prior = prediction_variance(y_final_mean_list_prior, y_final_var_list_prior).reshape(x_grid_mesh_shape)
    var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior).reshape(x_grid_mesh_shape)
    abs_var_diff = jnp.abs(var_prior - var_posterior)

    print("x_grid_mesh_shape: ", x_grid_mesh_shape)
    print("gp_mean_posterior shape: ", gp_mean_posterior.shape)
    print("abs_diff_gt_gp shape: ", abs_diff_gt_gp.shape)
    print("gp_mean_prior shape: ", gp_mean_prior.shape)
    print("abs_diff_prior shape: ", abs_diff_prior.shape)
    print("var_prior shape: ", var_prior.shape)
    print("var_posterior shape: ", var_posterior.shape)
    print("abs_var_diff shape: ", abs_var_diff.shape)
    print("u_values_gt shape: ", u_values_gt.shape)

    print("var_prior: ", var_prior)
    print("var_posterior: ", var_posterior)
    print("abs_var_diff: ", abs_var_diff)

    save_variables(added_text, u_values_gt=u_values_gt,
                   gp_mean_prior=gp_mean_prior,
                   abs_diff_prior=abs_diff_prior,
                   gp_mean_posterior=gp_mean_posterior,
                   abs_diff_gt_gp=abs_diff_gt_gp,
                   var_prior=var_prior,
                   var_posterior=var_posterior,
                   abs_var_diff=abs_var_diff,
                   add_text=added_text)


    plot_and_save_prediction_results_rd(u_values_gt,
                                     gp_mean_prior,
                                     abs_diff_prior,
                                    gp_mean_posterior,
                                    abs_diff_gt_gp,
                                    var_prior,
                                    var_posterior,
                                    abs_var_diff, added_text)

    plot_and_save_prediction_results_combine_rd(u_values_gt,
                                             gp_mean_prior,
                                             abs_diff_prior,
                                             gp_mean_posterior,
                                             abs_diff_gt_gp,
                                             var_prior,
                                             var_posterior,
                                             abs_var_diff, added_text)



    # # %%
    # u_values_gt = u_xt(X_plot_prediction)
    #
    # print("start plotting prediction results and saving as one image")
    #
    # plot_prediction_results_test(X_plot_prediction, u_values_gt, y_final_mean_list_prior, y_final_var_list_prior,
    #                         y_final_mean_list_posterior, y_final_var_list_posterior)
    # print("end plotting prediction results and saving as one image")
    # # %%
    #
    #
    #
    # # %%
    # print("start plotting prediction results and saving as single image")
    # plot_and_save_prediction_results(X_plot_prediction, u_values_gt, y_final_mean_list_prior, y_final_var_list_prior,
    #                                  y_final_mean_list_posterior, y_final_var_list_posterior)
    #
    # print("end plotting prediction results and saving as single image")
    # # %%