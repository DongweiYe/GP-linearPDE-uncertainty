import os
import jax
import datetime
import optax
from include.check_hyperparameters import check_hyperparamters
from include.heat2d import plot_u_f, f_xt, plot_u_f_pred, get_u_test_data_2d_qmc, plot_u_pred, u_xt
from include.mcmc_posterior import *
from include.init import ModelInitializer_2d
from include.plot_dist import plot_dist, plot_with_noise, plot_and_save_kde_histograms
from include.train import train_heat_equation_model_2d
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
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
epoch_pred = 100

noise_std = 0.1
prior_std = 0.04
prior_var = prior_std ** 2  # prior variance
max_samples = 3000
assumption_sigma = 0.05  # step size
k = 0.6
num_chains = 1

bw = 2
num_prior_samples = 400
learning_rate = 4e-2
test_num = 2 ** 4
number_u = 1  # xt
number_u_only_x =0
number_f = 2 ** 7
number_init = 2 ** 3
number_bound = 2 ** 3

param_text = "para"
optimizer_in_use = optax.adam
sample_num = 10
epochs = 1000
added_text = f'{number_u}&{number_u_only_x}&{number_f}&{number_init}&{number_bound}&{sample_num}&{epochs}&{noise_std}'
weight_decay = 1e-5
DEEP_FLAG = False
learning = f'{learning_rate}&{epochs}'

current_time = datetime.datetime.now().strftime("%m%d")
pred_mesh = 400

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

    x_prediction = jnp.linspace(0, 1, pred_mesh)
    t_prediction = jnp.linspace(0, 1, pred_mesh)

    X_prediction, T_prediction = jnp.meshgrid(x_prediction, t_prediction)

    X_plot_prediction = jnp.vstack([X_prediction.ravel(), T_prediction.ravel()]).T

    def save_individual_plot(data, title, cmap, vmin, vmax, filename):
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_heatmap(ax, data, title, cmap)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


    def plot_heatmap(ax, data, title, cmap, vmin, vmax):
        im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=20)
        ax.set_xlabel('x', fontsize=18)
        ax.set_ylabel('t', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=14)


    u_values_gt = u_xt(X_plot_prediction)
    save_individual_plot(u_values_gt, "ground_truth", 'GnBu', "test f 2*6")
