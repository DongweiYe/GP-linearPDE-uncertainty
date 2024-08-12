
import jax
import datetime
from include.heat2d import plot_u_pred, u_xt, compute_kuu, compute_kfu, compute_kuf
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
from include.mcmc_posterior import compute_K
from include.plot_dist import plot_dist
from include.plot_pred import plot_and_save_prediction_results
from include.plot_pred_test import plot_prediction_results_test
from include.train import train_heat_equation_model_2d

os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_enable_x64", True)

bw = 1
num_prior_samples = 500
current_time = datetime.datetime.now().strftime("%m%d")
learning_rate_pred = 0.1
epoch_pred = 10

text = "f32_chains1_k0.6_assumption0.5_prior_std0.3_noisestd0.1_init4_b4_0.09_k0.6_100_3404.pkl"
load_path = f"results/datas/trained_params/0812"


# %%

if __name__ == '__main__':
    # %%
    print('start inference')
    def generate_prior_samples(rng_key, num_samples, prior_mean, prior_cov):
        prior_samples = random.multivariate_normal(rng_key, mean=prior_mean.ravel(), cov=prior_cov,
                                                   shape=(num_samples,))
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

    Xu_without_noise = variables['Xu_without_noise']
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

    plt.rcParams["figure.figsize"] = (40, 10)

    fig1, axes1 = plt.subplots(2, number_u, figsize=(20, 10))

    for vague_points in range(number_u):
        ax = axes1[0, vague_points]
        data = posterior_samples_list[:, vague_points, 0]
        prior_data = prior_samples[:, vague_points * 2]
        ax.axvline(Xu_certain[vague_points, 0], color='tab:red', label='x GT', linestyle='--', linewidth=2)
        ax.axvline(Xu_noise[vague_points, 0], color='seagreen', label='x noised', linestyle=':', linewidth=2)
        sns.kdeplot(data, ax=ax, color='tab:blue', label='x denoised', bw_adjust=bw)
        sns.kdeplot(prior_data, ax=ax, color='tab:orange', label='x prior', linestyle='--')

        posterior_peak = find_kde_peak(data)
        ax.axvline(posterior_peak, color='tab:purple', linestyle='-.', linewidth=1, label='posterior peak')
        posterior_mean = jnp.mean(data)
        ax.axvline(posterior_mean, color='tab:cyan', linestyle='-', linewidth=2, label='posterior mean')

        gt_value = Xu_certain[vague_points, 0]
        noise_value = Xu_noise[vague_points, 0]
        values = [noise_value, posterior_peak, posterior_mean]
        labels = ['x noised', 'posterior peak', 'posterior mean']
        closest_label, distances = find_closest_to_gt(gt_value, values, labels)

        # ax.legend(loc='upper left')
        ax.set_xlabel(f'x_uncertain{vague_points}')
        # ax.text(0.5, -0.08, f'{closest_label}', transform=ax.transAxes, ha='center', va='top', fontsize=12, color='red')

    for vague_points in range(number_u):
        ax2 = axes1[1, vague_points]
        data1 = posterior_samples_list[:, vague_points, 1]
        prior_data1 = prior_samples[:, vague_points * 2 + 1]
        ax2.axvline(Xu_certain[vague_points, 1], color='tab:red', label='t GT', linestyle='--', linewidth=2)
        ax2.axvline(Xu_noise[vague_points, 1], color='seagreen', label='t noised', linestyle=':', linewidth=2)
        sns.kdeplot(data1, ax=ax2, color='tab:blue', label='t denoised', bw_adjust=bw)
        sns.kdeplot(prior_data1, ax=ax2, color='tab:orange', label='t prior', linestyle='--')

        posterior_peak1 = find_kde_peak(data1)
        ax2.axvline(posterior_peak1, color='tab:purple', linestyle='-.', linewidth=1, label='posterior peak')
        posterior_mean1 = jnp.mean(data1)
        ax2.axvline(posterior_mean1, color='tab:cyan', linestyle='-', linewidth=2, label='posterior mean')

        gt_value1 = Xu_certain[vague_points, 1]
        noise_value1 = Xu_noise[vague_points, 1]
        values1 = [noise_value1, posterior_peak1, posterior_mean1]
        labels1 = ['t noised', 'posterior peak', 'posterior mean']
        closest_label1, distances1 = find_closest_to_gt(gt_value1, values1, labels1)

        # ax2.legend(loc='upper left')
        ax2.set_xlabel(f't_uncertain{vague_points}')
        # ax2.text(0.5, -0.2, f' {closest_label1}', transform=ax2.transAxes, ha='center', va='top', fontsize=12,
        #          color='red')

    current_time = datetime.datetime.now().strftime("%M%S")
    fig1.savefig(
        f"LOADED_kdeplot_f{number_f}_chains{num_chains}_k{k}_assumption{assumption_sigma}_noisestd{noise_std}_{prior_var}_k{k}_{max_samples}_{current_time}.png",
        bbox_inches='tight')
    Xu_pred = jnp.mean(posterior_samples_list, axis=0)
    Xu_pred_map = posterior_samples_list[jnp.argmax(posterior_samples_list[:, -1])]
    plot_u_pred(Xu_without_noise, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred, prior_var, assumption_sigma, k,
                max_samples, learning, num_chains, number_f)

    fig2, axes2 = plt.subplots(2, number_u, figsize=(20, 10))

    fig2.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)

    for vague_points in range(number_u):
        ax = axes2[0, vague_points]
        posterior_data = posterior_samples_list[:, vague_points, 0]
        prior_data = prior_samples[:, vague_points * 2]

        ax.axvline(Xu_noise[vague_points, 0], color='seagreen', linestyle=':', linewidth=2, label='x noised')
        ax.hist(posterior_data, bins=30, density=True, alpha=0.6, color='tab:blue', label='x denoised')
        ax.hist(prior_data, bins=30, density=True, alpha=0.6, color='tab:orange', label='x prior')
        posterior_peak = find_kde_peak(posterior_data)
        # ax.axvline(posterior_peak, color='tab:purple', linestyle='-', linewidth=2, label='posterior peak')
        posterior_mean = jnp.mean(posterior_data)
        ax.axvline(posterior_mean, color='tab:cyan', linestyle='solid', linewidth=2, label='posterior mean')
        ax.axvline(Xu_certain[vague_points, 0], color='tab:red', linestyle='--', linewidth=2, label='x GT')

        values = [Xu_noise[vague_points, 0], posterior_peak, posterior_mean]
        labels = ['x noised', 'posterior peak', 'posterior mean']
        closest_label, _ = find_closest_to_gt(Xu_certain[vague_points, 0], values, labels)

        ax.set_xlabel(f'uncertain position {vague_points + 1}', fontsize=22)
        ax.set_ylabel('density', fontsize=22)
        # ax.text(0.5, -0.15, f'Closest to GT: {closest_label}', transform=ax.transAxes, ha='center', va='top',
        #         fontsize=14, color='blue')
        ax.tick_params(axis='both', which='major', labelsize=16)

    handles1, labels1 = [], []
    for ax in axes2[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels1:
                handles1.append(handle)
                labels1.append(label)

    fig2.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 0.93), fontsize=16, ncol=len(labels1))

    for vague_points in range(number_u):
        ax2 = axes2[1, vague_points]
        posterior_data1 = posterior_samples_list[:, vague_points, 1]
        prior_data1 = prior_samples[:, vague_points * 2 + 1]

        ax2.axvline(Xu_noise[vague_points, 1], color='seagreen', linestyle=':', linewidth=2, label='t noised')
        ax2.hist(posterior_data1, bins=30, density=True, alpha=0.6, color='tab:blue', label='t denoised')
        ax2.hist(prior_data1, bins=30, density=True, alpha=0.6, color='tab:orange', label='t prior')
        posterior_peak1 = find_kde_peak(posterior_data1)
        # ax2.axvline(posterior_peak1, color='tab:purple', linestyle='-', linewidth=2, label='posterior peak')
        posterior_mean1 = jnp.mean(posterior_data1)
        ax2.axvline(posterior_mean1, color='tab:cyan', linestyle='solid', linewidth=2, label='posterior mean')
        ax2.axvline(Xu_certain[vague_points, 1], color='tab:red', linestyle='--', linewidth=2, label='t GT')

        values1 = [Xu_noise[vague_points, 1], posterior_peak1, posterior_mean1]
        labels1 = ['t noised', 'posterior peak', 'posterior mean']
        closest_label1, _ = find_closest_to_gt(Xu_certain[vague_points, 1], values1, labels1)

        ax2.set_xlabel(f'uncertain time {vague_points + 1}', fontsize=22)
        ax2.set_ylabel('density', fontsize=22)
        # ax2.text(0.5, -0.15, f'Closest to GT: {closest_label1}', transform=ax2.transAxes, ha='center', va='top',
        #          fontsize=14, color='blue')
        ax2.tick_params(axis='both', which='major', labelsize=16)

    handles2, labels2 = [], []
    for ax2 in axes2[1]:
        for handle, label in zip(*ax2.get_legend_handles_labels()):
            if label not in labels2:
                handles2.append(handle)
                labels2.append(label)

    fig2.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 0.48), fontsize=16, ncol=len(labels2))

    fig2.savefig(
        f"LOADED_hist_f{number_f}_chains{num_chains}_k{k}_assumption{assumption_sigma}_noisestd{noise_std}_{prior_var}_k{k}_{max_samples}_{current_time}.png",
        bbox_inches='tight')

    added_text = f"Predction_f{number_f}_chains{num_chains}_k{k}_assumption{assumption_sigma}_noisestd{noise_std}_{prior_var}_k{k}_{max_samples}_{current_time}"

    Xu_pred_mean = jnp.mean(posterior_samples_list, axis=0)

    plot_u_pred(Xu_without_noise, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred_mean, prior_var, assumption_sigma, k,
                max_samples, learning, num_chains, number_f)
    plot_dist(Xu_without_noise, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred_mean, prior_var, assumption_sigma, k,
              max_samples, learning, num_chains, number_f, posterior_samples_list, prior_samples)

    print('end inference')

# %%
# # %%
    print("start prediction")
    x_prediction = jnp.linspace(0, 1, 40)
    t_prediction = jnp.linspace(0, 1, 40)

    X_prediction, T_prediction = jnp.meshgrid(x_prediction, t_prediction)

    X_plot_prediction = jnp.vstack([X_prediction.ravel(), T_prediction.ravel()]).T

    # y_final_mean_list_posterior = jnp.empty((0, X_plot_prediction.shape[0]))
    # y_final_var_list_posterior = jnp.empty((0, X_plot_prediction.shape[0]))
    #
    # y_final_mean_list_prior = jnp.empty((0, X_plot_prediction.shape[0]))
    # y_final_var_list_prior = jnp.empty((0, X_plot_prediction.shape[0]))

    y_final_mean_list_posterior = []
    y_final_var_list_posterior = []

    y_final_mean_list_prior = []
    y_final_var_list_prior = []


    def compute_joint_K(init, z_prior, Xcz, Xcg, x_star):
        Xuz = z_prior
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        params = init
        K = compute_K(init, z_prior, Xcz, Xcg)
        lengthscale_x = params[0][1][0].item()
        lengthscale_t = params[0][1][1].item()

        k_zz_u_star = compute_kuu(Xuz, x_star, params_kuu)
        k_zz_c_star = compute_kuu(Xcz, x_star, params_kuu)
        k_gz_c_star = compute_kfu(Xcg, x_star, params, lengthscale_x, lengthscale_t)

        k_x_star = jnp.vstack((k_zz_u_star, k_zz_c_star, k_gz_c_star))

        k_x_star_x_star = compute_kuu(x_star, x_star, params_kuu)

        joint_K = jnp.block([[K, k_x_star], [k_x_star.T, k_x_star_x_star]])

        return joint_K


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

    def gp_predict(init, z_prior, Xcz, Xcg, y, x_star):
        print("Starting gp_predict function")

        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        params = init
        K = compute_K(init, z_prior, Xcz, Xcg)
        print("Computed K matrix")

        lengthscale_x = params[0][1][0].item()
        lengthscale_t = params[0][1][1].item()

        k_zz_u_star = compute_kuu(z_prior, x_star, params_kuu)
        k_zz_c_star = compute_kuu(Xcz, x_star, params_kuu)
        k_gz_c_star = compute_kfu(Xcg, x_star, params, lengthscale_x, lengthscale_t)
        # k_gz_c_star_T = compute_kuf(x_star, Xcg, params)
        k_x_star = jnp.vstack((k_zz_u_star, k_zz_c_star, k_gz_c_star))
        k_x_star_x_star = compute_kuu(x_star, x_star, params_kuu)
        # print("jnp.allclose(k_gz_c_star.T, k_gz_c_star_T)", jnp.allclose(k_gz_c_star.T, k_gz_c_star_T))

        # # CPU
        # K_cpu = jax.device_put(K + 1e-6 * jnp.eye(K.shape[0]), device=jax.devices("cpu")[0])
        # y_cpu = jax.device_put(y.reshape(-1, 1), device=jax.devices("cpu")[0])
        # k_x_star_cpu = jax.device_put(k_x_star, device=jax.devices("cpu")[0])
        # print("Moved data to CPU")
        # K_inv_y = la.solve(K_cpu, y_cpu, assume_a='pos')
        # K_inv_k_x_star = la.solve(K_cpu, k_x_star_cpu, assume_a='pos')
        # mu_star_cpu = jnp.dot(k_x_star_cpu.T, K_inv_y)
        # sigma_star_cpu = k_x_star_x_star - jnp.einsum('ij,ij->i', k_x_star_cpu.T, K_inv_k_x_star.T).reshape(-1, 1)
        # ## GPU
        # mu_star_gpu = jax.device_put(mu_star_cpu, device=jax.devices("gpu")[0])
        # sigma_star_gpu = jax.device_put(sigma_star_cpu, device=jax.devices("gpu")[0])
        # print("Moved results back to GPU")
        # # try:
        # #     sigma_star_gpu = jax.device_put(sigma_star_cpu, device=jax.devices("gpu")[0])
        # # except jaxlib.xla_extension.XlaRuntimeError:
        # #     print("GPU 内存不足，无法将 sigma_star 传输到 GPU")
        # #     raise
        # del K_cpu, y_cpu, k_x_star_cpu, K_inv_y, K_inv_k_x_star, mu_star_cpu, sigma_star_cpu
        #

        # # only GPU
        K_inv_y = la.solve(K, y, assume_a='pos')
        K_inv_k_x_star = la.solve(K, k_x_star, assume_a='pos')
        mu_star_gpu = jnp.dot(k_x_star.T, K_inv_y)
        sigma_star_gpu = k_x_star_x_star - jnp.einsum('ij,ij->i', k_x_star.T, K_inv_k_x_star.T).reshape(-1, 1)
        del K, k_x_star, K_inv_y, K_inv_k_x_star
        gc.collect()
        return mu_star_gpu.flatten(), sigma_star_gpu


    for i in range(posterior_samples_list.shape[0]):
        Xu_sample = posterior_samples_list[i, :, :]

        param_iter, _, _, _ = train_heat_equation_model_2d(param_iter,
                                                           Xu_sample,
                                                           Xu_fixed,
                                                           Xf,
                                                           Y.shape[0],
                                                           Y, epoch_pred,
                                                           learning_rate_pred,
                                                           optimizer_in_use)

        lengthscale = param_iter[-1][1]
        sigma = param_iter[-1][0]

        y_final_mean, y_final_var = gp_predict(param_iter, Xu_sample, Xu_fixed, Xf, Y, X_plot_prediction)
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

    y_final_mean_list_posterior = jnp.array(y_final_mean_list_posterior)
    y_final_var_list_posterior = jnp.array(y_final_var_list_posterior)

    y_final_mean_list_posterior = jnp.vstack(y_final_mean_list_posterior)
    y_final_var_list_posterior = jnp.vstack(y_final_var_list_posterior)

    print("Prediction mean shape: ", y_final_mean_list_posterior.shape)
    print("Prediction variance shape: ", y_final_var_list_posterior.shape)

    prior_samples_reshaped = prior_samples.reshape(prior_samples.shape[0], -1, 2)
    for i in range(prior_samples_reshaped.shape[0]):
        Xu_sample_prior = prior_samples_reshaped[i, :, :]

        param_iter, _, _, _ = train_heat_equation_model_2d(param_iter,
                                                           Xu_sample_prior,
                                                           Xu_fixed,
                                                           Xf,
                                                           Y.shape[0],
                                                           Y, epoch_pred,
                                                           learning_rate_pred,
                                                           optimizer_in_use)

        lengthscale = param_iter[-1][1]
        sigma = param_iter[-1][0]

        y_final_mean_prior, y_final_var_prior = gp_predict(param_iter, Xu_sample_prior, Xu_fixed, Xf, Y,
                                                           X_plot_prediction)

        # y_final_mean_list_prior = jnp.vstack((y_final_mean_list_prior, y_final_mean_prior.reshape(1, -1)))
        # y_final_var_list_prior = jnp.vstack((y_final_var_list_prior, y_final_var_prior.reshape(1, -1)))
        y_final_mean_list_prior.append(y_final_mean_prior.reshape(1, -1))
        y_final_var_list_prior.append(y_final_var_prior.reshape(1, -1))

        del Xu_sample_prior, y_final_mean_prior, y_final_var_prior

        gc.collect()
        jax.clear_caches()
        print("prior memory cleaned up after iteration", i)

    y_final_mean_list_prior = jnp.array(y_final_mean_list_prior)
    y_final_var_list_prior = jnp.array(y_final_var_list_prior)

    y_final_mean_list_prior = jnp.vstack(y_final_mean_list_prior)
    y_final_var_list_prior = jnp.vstack(y_final_var_list_prior)

    def prediction_mean(ypred_list):
        return jnp.mean(ypred_list, axis=0)

    def prediction_variance(ypred_list, yvar_list):
        ymean_var = jnp.var(ypred_list, axis=0)
        yvar_mean = jnp.mean(yvar_list, axis=0)

        return ymean_var + yvar_mean

    y_final_mean_posterior = prediction_mean(y_final_mean_list_posterior)
    y_final_var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior)
    y_final_mean_prior = prediction_mean(y_final_mean_list_prior)
    y_final_var_prior = prediction_variance(y_final_mean_list_prior, y_final_var_list_prior)
    print("posterior Prediction mean shape: ", y_final_mean_posterior.shape)
    print("posterior Prediction variance shape: ", y_final_var_posterior.shape)
    print("prior Prediction mean shape: ", y_final_mean_prior.shape)
    print("prior Prediction variance shape: ", y_final_var_prior.shape)
    print("-------------------end prediction-------------------")
    def save_variables(added_text, **variables):
        root_folder = "."
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        filename = f"Pred_{added_text}.pkl"
        file_path = os.path.join(root_folder, filename)

        with open(file_path, 'wb') as f:
            pickle.dump(variables, f)
        print(f"Variables saved to {file_path}")


    u_values_gt = u_xt(X_plot_prediction)
    save_variables(added_text, X_plot_prediction=X_plot_prediction, u_values_gt=u_values_gt,
                   y_final_mean_list_prior=y_final_mean_list_prior, y_final_var_list_prior=y_final_var_list_prior,
                   y_final_mean_list_posterior=y_final_mean_list_posterior, y_final_var_list_posterior=y_final_var_list_posterior)

    plot_and_save_prediction_results(X_plot_prediction, u_values_gt, y_final_mean_list_prior, y_final_var_list_prior,
                                     y_final_mean_list_posterior, y_final_var_list_posterior)

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