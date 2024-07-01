import os
import jax
import datetime
import optax
from include.check_hyperparameters import check_hyperparamters
from include.heat2d import plot_u_f
from include.mcmc_posterior import *
from include.init import ModelInitializer_2d
from include.train import train_heat_equation_model_2d
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from scipy.optimize import minimize


os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_enable_x64", True)

## modified
# 1: generate data through QMC

# %%
noise_std = 1e-1
prior_var = 0.1#k * (noise_std ** 2)
number_u = 2**1
number_f = 2**5
number_init = 2**2
number_bound = 2**2
max_samples = 500
assumption_variance = 0.1
num_mcmc = 200
k = 0.2

param_text = "para"
optimizer_in_use = optax.adam
sample_num = 10
epochs = 1000
added_text = f'{number_u}&{number_f}&{number_init}&{number_bound}&{sample_num}&{epochs}&{noise_std}'
learning_rate = 1e-3
weight_decay = 1e-5
DEEP_FLAG = False
num_warmup = 100
# %%
#TODO
# posterior MAP
# reconstruct fixed

def initialize_params_2d(sigma_init, lengthscale_init):
    # Initialize RBF kernel hyperparameters
    sigma = jnp.array([sigma_init])
    lengthscale = lengthscale_init

    params = (
        (sigma, lengthscale),
    )

    return params

if __name__ == '__main__':
    print("noise_std:", noise_std, "\n")
    print("prior_var:", prior_var, "\n")
    print("number_u:", number_u, "\n")
    print("number_f:", number_f, "\n")
    print("number_init:", number_init, "\n")
    print("number_bound:", number_bound, "\n")
    print("max_samples:", max_samples, "\n")
    print("assumption_variance:", assumption_variance, "\n")
    print("num_mcmc:", num_mcmc, "\n")
    print("k:", k, "\n")
    print("param_text:", param_text, "\n")
    print("optimizer_in_use:", optimizer_in_use, "\n")
    print("sample_num:", sample_num, "\n")
    print("epochs:", epochs, "\n")
    print("added_text:", added_text, "\n")
    print("learning_rate:", learning_rate, "\n")
    print("weight_decay:", weight_decay, "\n")
    print("DEEP_FLAG:", DEEP_FLAG, "\n")
    print("num_warmup:", num_warmup, "\n")

    model_initializer = ModelInitializer_2d(number_u=number_u, number_f=number_f, sample_num=sample_num,
                                            number_init=number_init, number_bound=number_bound, noise_std=noise_std)
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
    plot_u_f(Xu_without_noise, Xf, Xu_noise)
    """
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
    
    # param_iter = initialize_params_2d(0.44709868, jnp.array([0.13204616, 0.24606263]))
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
    #MAP
    # initial_guess = jnp.ravel(Xu_noise)
    # result = minimize(neg_log_posterior, initial_guess, args=(Xu_fixed, Xf, Y, Xu_noise, prior_var*jnp.eye(2), param_iter),
    #                   method='L-BFGS-B')
    # map_estimate = result.x.reshape(Xu_noise.shape)
    #
    # print("MAP Estimate:\n", map_estimate)



    print(f"assumption_variance={assumption_variance}")
    max_samples = num_mcmc
    rng_key = jax.random.PRNGKey(9)
    #posterior_samples = Metropolis_Hasting(max_samples, assumption_variance, Xu_noise, jnp.eye(2*number_u)*prior_var, param_iter, Xu_fixed, Xf, Y)
    posterior_samples = metropolis_hasting(rng_key, max_samples, assumption_variance, Xu_noise, jnp.eye(2)*prior_var, param_iter, Xu_fixed, Xf, Y, k)
    num_samples = Xu_noise.shape[0]
    z_uncertain_means = []

    # posterior_samples_list = np.mean(posterior_samples, axis=1)
    posterior_samples_list = posterior_samples.reshape(-1, *Xu_noise.shape)
    print("posterior_samples.shape:", posterior_samples.shape)
    print("posterior_samples_list shape:", posterior_samples_list.shape)


    plt.rcParams["figure.figsize"] = (40, 10)

    fig1, axes1 = plt.subplots(1, 2*number_u)
    for vague_points in range(number_u):
        # fig = plt.figure()
        ax = axes1[vague_points]
        data = posterior_samples_list[:, vague_points, 0]
        ax.axvline(Xu_certain[vague_points, 0], color='tab:red',label='x GT')
        ax.axvline(Xu_noise[vague_points, 0], color='tab:green',label='x noised')
        # plt.axvline(posterior_samples_list[vague_points], color='tab:blue',label='denoised')
        sns.kdeplot(data, ax=ax, color='tab:blue',label='x denoised')
        ax.legend()
        ax.set_xlabel(f'x_uncertain{vague_points}')
        current_time = datetime.datetime.now().strftime("%M%S")
        # ax.savefig(f"kdeplot_x_{prior_var}_k{k}_{assumption_variance}_{vague_points}_{current_time}.png",bbox_inches='tight')
        # plt.close()

    # fig2, axes2 = plt.subplots(1, number_u)
    for vague_points in range(number_u):
        ax2 = axes1[vague_points+number_u]
        data1 =  posterior_samples_list[:, vague_points, 1]
        ax2.axvline(Xu_certain[vague_points, 1], color='tab:red',label='t GT')
        ax2.axvline(Xu_noise[vague_points, 1], color='tab:green',label='t noised')
        # plt.axvline(posterior_samples_list[vague_points], color='tab:blue',label='denoised')
        sns.kdeplot(data1, ax=ax2, color='tab:blue',label='t denoised')
        ax2.legend()
        ax2.set_xlabel(f't_uncertain{vague_points}')
        current_time = datetime.datetime.now().strftime("%M%S")
        # plt.savefig(f"kdeplot_t_{prior_var}_k{k}_{assumption_variance}_{vague_points}_{current_time}.png",bbox_inches='tight')
        # plt.close()
    current_time = datetime.datetime.now().strftime("%M%S")
    fig1.savefig(f"kdeplot_xt_{num_mcmc}_{prior_var}_k{k}_{assumption_variance}_{max_samples}_{current_time}.png",
                bbox_inches='tight')
                """
    # mcmc = run_mcmc(Xu_fixed, Xf, Y, Xu_noise, prior_var*jnp.eye(2), param_iter, num_samples=num_mcmc,
    #                 num_warmup=num_warmup)
    # posterior_samples = mcmc.get_samples()
    #
    # z_uncertain_samples = posterior_samples['z_uncertain_flat'].reshape(-1, *Xu_noise.shape)
    # print("z_uncertain_samples[:, 1, 0]", z_uncertain_samples[:, 1, 0])
    # print("posterior_samples['z_uncertain_flat'] shape:", posterior_samples['z_uncertain_flat'].shape)
    # print("z_uncertain_samples shape:", z_uncertain_samples.shape)
    # fig1, axes1 = plt.subplots(1, number_u)
    # for i in range(number_u):
    #     data = z_uncertain_samples[:, i, 0]
    #     ax = axes1[i]
    #     sns.kdeplot(data, ax=ax, color='tab:blue', label=f'z_uncertain{i}_0 Posterior')
    #     ax.axvline(Xu_certain[i, 0], color='tab:red', label='Xu True Value')
    #     ax.axvline(Xu_noise[i, 0], color='tab:green', label='Xu Noise Value')
    #     # ax.axvline(map_estimate[i, 0], color='tab:orange', label='MAP Estimate')
    #     ax.set_xlabel(f'z_uncertain{i}_0')
    #     ax.set_ylabel('Density')
    #     ax.set_title(f'Posterior Distribution of z_uncertain{i}_0')
    #     ax.legend()
    #
    # # fig1.tight_layout()
    # current_time = datetime.datetime.now().strftime("%M%S")
    # fig1.savefig(
    #     f"x_lr{learning_rate}_{added_text}_priorvar{prior_var}_noisestd{noise_std}_samwarp_{num_mcmc}&{num_warmup}_{current_time}.pdf",
    #     format='pdf')
    #
    #
    # fig2, axes2 = plt.subplots(1, number_u)
    # for i in range(number_u):
    #     data = z_uncertain_samples[:, i, 1]
    #     ax = axes2[i]
    #     sns.kdeplot(data, ax=ax, color='tab:blue', label=f'z_uncertain{i}_1 Posterior')
    #     ax.axvline(Xu_certain[i, 1], color='tab:red', label='Xu True Value')
    #     ax.axvline(Xu_noise[i, 1], color='tab:green', label='Xu Noise Value')
    #     # ax.axvline(map_estimate[i, 1], color='tab:orange', label='MAP Estimate')
    #     ax.set_xlabel(f'z_uncertain{i}_1')
    #     ax.set_ylabel('Density')
    #     ax.set_title(f'Posterior Distribution of z_uncertain{i}_1')
    #     ax.legend()
    #
    # current_time = datetime.datetime.now().strftime("%M%S")
    # fig2.savefig(f"t_{param_text}{learning_rate}_{added_text}_priorvar{prior_var}_noisestd{noise_std}_samwarp_{num_mcmc}&{num_warmup}_{current_time}.pdf",format='pdf')
    #




    # print("#############################################")
    # print("Posterior samples keys:", posterior_samples.keys())
    # print("#############################################")
    """
    for i in range(num_samples):
        param_name = f'z_uncertain{i}'
        if param_name in posterior_samples:
            data = posterior_samples[param_name]
            plt.figure(figsize=(20, 16))
            # plt.hist(data, bins=30, alpha=0.7, label=f'{param_name} posterior')
            # sns.kdeplot(data, color='tab:blue', label=f'{param_name} Posterior')
            plt.xlabel(param_name)
            plt.ylabel('Density')
            plt.title(f'Posterior Distribution of {param_name}')
            plt.axvline(Xu_certain[i, 0], color='tab:red', label='Xu True Value')
            plt.axvline(Xu_noise[i, 0], color='tab:green', label='Xu Noise Value')
            # plt.axvline(data[0], color='tab:orange', label='Posterior')
            sns.kdeplot(data[0], color='tab:blue', label=f'{param_name} Posterior')
            # sns.kdeplot(posterior_samples['z_uncertain' + str(i)], color='tab:blue', label='Posterior KDE')
            current_time = datetime.datetime.now().strftime("%M%S")
            plt.legend()
            # plt.savefig(f"hist_posterior_{param_name}.pdf", format='pdf')
            plt.savefig(f"{param_name}_lr{learning_rate}_{added_text}_priorvar{prior_var}_noisestd{noise_std}_samwarp_{num_samples}&{num_warmup}_{current_time}.pdf", format='pdf', bbox_inches='tight')
            plt.close()
        else:
            print(f'Key {param_name} not found in posterior_samples.')
"""
"""
    # posterior_samples_list = z_uncertain_means
    # for vague_points in range(1):
    #     fig = plt.figure()
    #     plt.axvline(Xu[vague_points, 0], color='tab:red')
    #     plt.axvline(Xu_noise[vague_points, 0], color='tab:green')
    #     sns.kdeplot(posterior_samples_list[vague_points, :], color='tab:blue')
    #     plt.savefig(f"kdeplot_1_{vague_points}.pdf", format='pdf')
"""

