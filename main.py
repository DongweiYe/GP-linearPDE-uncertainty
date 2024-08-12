import os
import jax
import datetime
import optax
from include.check_hyperparameters import check_hyperparamters
from include.heat2d import plot_u_f, f_xt, plot_u_f_pred, get_u_test_data_2d_qmc, plot_u_pred
from include.mcmc_posterior import *
from include.init import ModelInitializer_2d
from include.plot_dist import plot_dist
from include.train import train_heat_equation_model_2d
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from scipy.optimize import minimize
from jax import random
from scipy.stats import gaussian_kde
from functools import partial
import pickle
import os
import jax.scipy.linalg as la

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


# TODO
# 1: reaction_diffusion
# 3: prediction
# 4: 画分布图
#检查是不是画一个图,画出一个图多分布

#跑出结果来
# %%
learning_rate_pred = 0.01
epoch_pred= 100
noise_std = 0.1
prior_std = 0.3
prior_var = prior_std**2# prior variance
max_samples = 100
assumption_sigma = 0.5 # step size
k = 0.6
num_chains = 1

bw=2
num_prior_samples = 5000
learning_rate = 4e-2
test_num = 2**4
number_u = 2**2 # 2种情况都有 x=4 xt=4
number_u_only_x = 0
number_f = 2**5
number_init = 2**2
number_bound = 2**2

param_text = "para"
optimizer_in_use = optax.adam
sample_num = 10
epochs = 1000
added_text = f'{number_u}&{number_u_only_x}&{number_f}&{number_init}&{number_bound}&{sample_num}&{epochs}&{noise_std}'
weight_decay = 1e-5
DEEP_FLAG = False
learning = f'{learning_rate}&{epochs}'

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
                                            number_init=number_init, number_bound=number_bound, noise_std=noise_std, number_u_only_x=number_u_only_x)
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

    Xu = model_initializer.Xu_with_noise
    Xu_without_noise = model_initializer.Xu_without_noise
    Yu = model_initializer.Yu
    print("Xu:", Xu)
    print("Yu:", Yu)

    Xf = model_initializer.Xf
    yf = model_initializer.yf
    print("Xf:", Xf)
    print("yf:", yf)

    Y = model_initializer.Y
    X_with_noise = model_initializer.X_with_noise

    X_without_noise = model_initializer.X_without_noise
    print("Y:", Y)
    print("X:", X_with_noise)

    xtest = model_initializer.xtest
    ytest = model_initializer.ytest

    number_u = model_initializer.number_u
    number_f = model_initializer.number_f
    number_init = model_initializer.number_init
    number_bound = model_initializer.number_bound
    number_Y = model_initializer.number_Y

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

    # check_hyperparamters(init, param_iter, f_xt, Xu_fixed, Yu_fixed, Xf, yf)

    # posterior_samples_list = Metropolis_Hasting(max_samples, assumption_variance, Xu_noise,
    #                                             jnp.eye(2*number_u)*prior_std**2, param_iter, Xu_fixed, Xf, Y)
    # trace = posterior_inference_mcmc(Xu_noise, jnp.eye(2)*prior_var, param_iter, Xu_fixed, Xf, Y)
    # trace = posterior_numpyro(Xu_noise, jnp.eye(2)*prior_var, param_iter, Xu_fixed, Xf, Y)

    # posterior_samples = jnp.squeeze(trace.get_values('z_uncertain', combine=True))
    # print(posterior_samples.shape)
    # print(posterior_samples)

    # z_uncertain_means = []
    # for i in range(num_samples):
    #     param_name = f'z_uncertain{i}'
    #     z_uncertain_mean = np.mean(posterior_samples[param_name], axis=0)
    #     print(f"{i}:_z_uncertain_mean={z_uncertain_mean}")
    #     z_uncertain_means.append(z_uncertain_mean)

    # trace = run_mcmc(Xu_fixed, Xf, Y, Xu_noise, jnp.eye(2) * prior_var, param_iter, num_samples=num_samples, num_warmup=num_warmup)
    # posterior_samples = trace
    # print(posterior_samples)

# %%
    print('start inference')
    # MAP
    # initial_guess = jnp.ravel(Xu_noise)
    # result = minimize(neg_log_posterior, initial_guess, args=(Xu_fixed, Xf, Y, Xu_noise, prior_var*jnp.eye(2), param_iter),
    #                   method='L-BFGS-B')
    # map_estimate = result.x.reshape(Xu_noise.shape)
    #
    # print("MAP Estimate:\n", map_estimate)
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
    prior_samples = generate_prior_samples(prior_key, num_prior_samples, Xu_noise, prior_cov_flat)


    print(f"assumption_sigma={assumption_sigma}")
    rng_key = jax.random.PRNGKey(42)
    #####posterior_samples = Metropolis_Hasting(max_samples, assumption_variance, Xu_noise, jnp.eye(2*number_u)*prior_var, param_iter, Xu_fixed, Xf, Y)
    #posterior_samples = single_component_metropolis_hasting(rng_key, max_samples, assumption_sigma, Xu_noise, jnp.eye(2)*prior_var, param_iter, Xu_fixed, Xf, Y, k)
   #### posterior_samples = gibbs_sampling(rng_key, max_samples, Xu_noise, jnp.eye(2)*prior_var, param_iter, Xu_fixed, Xf, Y, k)
    all_chains_samples = []

    for chain_id in range(num_chains):
        rng_key, chain_key = random.split(rng_key)
        chain_samples = single_component_metropolis_hasting(chain_key, max_samples, assumption_sigma, Xu_noise,
                                                            jnp.eye(2) * prior_var, param_iter, Xu_fixed, Xf, Y, k, number_u_only_x)
        all_chains_samples.append(chain_samples)

    all_chains_samples = jnp.array(all_chains_samples)
    num_samples = Xu_noise.shape[0]
    z_uncertain_means = []

    #burn_in = 30
    #posterior_samples = posterior_samples[burn_in:]
    # posterior_samples_list = np.mean(posterior_samples, axis=1)

    posterior_samples_list = jnp.concatenate(all_chains_samples, axis=0)
    posterior_samples_list = posterior_samples_list.reshape(-1, *Xu_noise.shape)
    #posterior_samples_list = posterior_samples.reshape(-1, *Xu_noise.shape)
    # print("posterior_samples.shape:", posterior_samples.shape)
    # print("posterior_samples:", posterior_samples)
    print("posterior_samples_list shape:", posterior_samples_list.shape)
    print("posterior_samples_list:", posterior_samples_list)
    print("Xu_certain:", Xu_certain)
    print("Xu_noise:", Xu_noise)

    plt.rcParams["figure.figsize"] = (40, 10)
    plt.rcParams.update({'font.size': 18})

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
        #ax.axvline(posterior_peak, color='tab:purple', linestyle='-.', linewidth=1, label='posterior peak')
        posterior_mean = jnp.mean(data)
        ax.axvline(posterior_mean, color='tab:cyan', linestyle='-', linewidth=2, label='posterior mean')

        gt_value = Xu_certain[vague_points, 0]
        noise_value = Xu_noise[vague_points, 0]
        # values = [noise_value, posterior_mean]
        # labels = ['x noised', 'posterior mean']
        # closest_label, distances = find_closest_to_gt(gt_value, values, labels)

       # ax.legend(loc='upper left')
        ax.set_xlabel(f'x_uncertain{vague_points}')
        # ax.text(0.5, -0.08 , f'{closest_label}', transform=ax.transAxes, ha='center', va='top', fontsize=12, color='red')

    for vague_points in range(number_u):
        ax2 = axes1[1, vague_points]
        data1 = posterior_samples_list[:, vague_points, 1]
        prior_data1 = prior_samples[:, vague_points * 2 + 1]
        ax2.axvline(Xu_certain[vague_points, 1], color='tab:red', label='t GT', linestyle='--', linewidth=2)
        ax2.axvline(Xu_noise[vague_points, 1], color='seagreen', label='t noised', linestyle=':', linewidth=2)
        sns.kdeplot(data1, ax=ax2, color='tab:blue', label='t denoised', bw_adjust=bw)
        sns.kdeplot(prior_data1, ax=ax2, color='tab:orange', label='t prior', linestyle='--')

        posterior_peak1 = find_kde_peak(data1)
        #ax2.axvline(posterior_peak1, color='tab:purple', linestyle='-.', linewidth=1, label='posterior peak')
        posterior_mean1 = jnp.mean(data1)
        ax2.axvline(posterior_mean1, color='tab:cyan', linestyle='-', linewidth=2, label='posterior mean')

        gt_value1 = Xu_certain[vague_points, 1]
        noise_value1 = Xu_noise[vague_points, 1]
        # values1 = [noise_value1, posterior_mean1]
        # labels1 = ['t noised', 'posterior mean']
        # closest_label1, distances1 = find_closest_to_gt(gt_value1, values1, labels1)

        #ax2.legend(loc='upper left')
        ax2.set_xlabel(f't_uncertain{vague_points}')
        # ax2.text(0.5, -0.2, f' {closest_label1}', transform=ax2.transAxes, ha='center', va='top', fontsize=12,
        #          color='red')

    current_time = datetime.datetime.now().strftime("%M%S")
    fig1.savefig(
        f"kdeplot_f{number_f}_chains{num_chains}_k{k}_assumption{assumption_sigma}_prior_std{prior_std}_noisestd{noise_std}_init{number_init}_b{number_bound}_{prior_var}_k{k}_{max_samples}_{current_time}.png",
        bbox_inches='tight')


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

        # values = [Xu_noise[vague_points, 0], posterior_peak, posterior_mean]
        # labels = ['x noised', 'posterior peak', 'posterior mean']
        # closest_label, _ = find_closest_to_gt(Xu_certain[vague_points, 0], values, labels)

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

        # values1 = [Xu_noise[vague_points, 1], posterior_peak1, posterior_mean1]
        # labels1 = ['t noised', 'posterior mean']
        #closest_label1, _ = find_closest_to_gt(Xu_certain[vague_points, 1], values1, labels1)

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
            f"hist_f{number_f}_chains{num_chains}_k{k}_assumption{assumption_sigma}_prior_std{prior_std}_noisestd{noise_std}_init{number_init}_b{number_bound}_{prior_var}_k{k}_{max_samples}_{current_time}.png",
            bbox_inches='tight')
    added_text =  f"f{number_f}_chains{num_chains}_k{k}_assumption{assumption_sigma}_prior_std{prior_std}_noisestd{noise_std}_init{number_init}_b{number_bound}_{prior_var}_k{k}_{max_samples}_{current_time}"



    Xu_pred_mean = jnp.mean(posterior_samples_list, axis=0)
    # Xu_pred_map = posterior_samples_list[jnp.argmax(posterior_samples_list[:, -1])]
    plot_u_pred(Xu_without_noise, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred_mean, prior_var,assumption_sigma,k,max_samples,learning,num_chains,number_f)
    plot_dist(Xu_without_noise, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred_mean, prior_var,assumption_sigma,k,max_samples,learning,num_chains,number_f,posterior_samples_list, prior_samples)


    save_variables(added_text, Xu_without_noise=Xu_without_noise, Xu_certain=Xu_certain, Xf=Xf, Xu_noise=Xu_noise,
                   noise_std=noise_std, Xu_pred=Xu_pred_mean, prior_var=prior_var, assumption_sigma=assumption_sigma,
                   k=k, max_samples=max_samples, learning=learning, num_chains=num_chains, number_f=number_f,
                   posterior_samples_list=posterior_samples_list, prior_samples=prior_samples, Y=Y,
                   param_iter=param_iter, Xu_fixed=Xu_fixed, epochs=epochs,
                   learning_rate=learning_rate,
                   optimizer_in_use=optimizer_in_use)

# %%
# # %%
    """
    print("start prediction")
    x_prediction = jnp.linspace(0, 1, 100)
    t_prediction = jnp.linspace(0, 1, 100)

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

        k_zz_u_star = compute_kuu(Xuz, x_star, params_kuu )
        k_zz_c_star = compute_kuu(Xcz, x_star, params_kuu )
        k_gz_c_star = compute_kfu(Xcg, x_star, params)

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
    #     K_inv = la.inv(K + 1e-6 * jnp.eye(K.shape[0]))  # add jitter for numerical stability
    #
    #     y = y.reshape(-1, 1)
    #     mu_star = k_x_star.T @ K_inv @ y
    #
    #     sigma_star = k_x_star_x_star - k_x_star.T @ K_inv @ k_x_star
    #
    #     return mu_star.flatten(), sigma_star
    def gp_predict(init, z_prior, Xcz, Xcg, y, x_star):
        print("Starting gp_predict function")

        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        params = init
        K = compute_K(init, z_prior, Xcz, Xcg)
        print("Computed K matrix")

        k_zz_u_star = compute_kuu(z_prior, x_star, params_kuu)
        k_zz_c_star = compute_kuu(Xcz, x_star, params_kuu)
        k_gz_c_star = compute_kfu(Xcg, x_star, params)
        k_gz_c_star_T = compute_kuf(x_star, Xcg, params)
        k_x_star = jnp.vstack((k_zz_u_star, k_zz_c_star, k_gz_c_star))
        print("k_gz_c_star", k_gz_c_star.T)

        k_x_star_x_star = compute_kuu(x_star, x_star, params_kuu)
        print("k_gz_c_star_T", k_gz_c_star_T)
        print("jnp.allclose(k_gz_c_star.T, k_gz_c_star_T)", jnp.allclose(k_gz_c_star.T, k_gz_c_star_T))

        # CPU
        K_cpu = jax.device_put(K + 1e-6 * jnp.eye(K.shape[0]), device=jax.devices("cpu")[0])
        y_cpu = jax.device_put(y.reshape(-1, 1), device=jax.devices("cpu")[0])
        k_x_star_cpu = jax.device_put(k_x_star, device=jax.devices("cpu")[0])
        print("Moved data to CPU")

        K_inv_y = la.solve(K_cpu, y_cpu, assume_a='pos')
        K_inv_k_x_star = la.solve(K_cpu, k_x_star_cpu, assume_a='pos')
        print("Solved linear equations on CPU")

        mu_star_cpu = jnp.dot(k_x_star_cpu.T, K_inv_y)
        print("Computed mu_star on CPU")

        sigma_star_cpu = k_x_star_x_star - jnp.einsum('ij,ij->i', k_x_star_cpu.T, K_inv_k_x_star.T).reshape(-1, 1)
        print("Computed sigma_star on CPU")

        # GPU
        mu_star_gpu = jax.device_put(mu_star_cpu, device=jax.devices("gpu")[0])
        sigma_star_gpu = jax.device_put(sigma_star_cpu, device=jax.devices("gpu")[0])
        print("Moved results back to GPU")
        # # GPU
        # mu_star = jax.device_put(mu_star, device=jax.devices("gpu")[0])
        # k_x_star = jax.device_put(k_x_star, device=jax.devices("gpu")[0])
        # K_inv_k_x_star = jax.device_put(K_inv_k_x_star, device=jax.devices("gpu")[0])
        # print("Moved results back to GPU")
        #
        # sigma_star = k_x_star_x_star - jnp.einsum('ij,ij->i', k_x_star.T, K_inv_k_x_star.T).reshape(-1, 1)
        # print("Computed sigma_star")

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

        # y_final_mean_list_posterior = jnp.vstack((y_final_mean_list_posterior, y_final_mean.T))
        # y_final_var_list_posterior = jnp.vstack((y_final_var_list_posterior, y_final_var.T))
        y_final_mean_list_posterior.append(y_final_mean.T)
        y_final_var_list_posterior.append(y_final_var.T)

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
        
    """

