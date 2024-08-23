
import jax
import datetime
from include.heat2d import plot_u_pred
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from jax import random
from scipy.stats import gaussian_kde
import pickle
import os

from include.plot_dist import plot_dist

os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_enable_x64", True)

bw = 1
num_prior_samples = 10000
current_time = datetime.datetime.now().strftime("%m%d")


text = "f32_chains1_k0.6_assumption0.008_prior_std0.1_noisestd0.08_init4_b4_0.010000000000000002_k0.6_20000_2353.pkl"
load_path = f"results/datas/trained_params/{current_time}"


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
    Y= variables['Y']
    param_iter = variables['param_iter']
    Xu_fixed = variables['Xu_fixed']
    epochs = variables['epochs']
    learning_rate = variables['learning_rate']
    optimizer_in_use = variables['optimizer_in_use']

    number_u = Xu_noise.shape[0]

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
        ax.text(0.5, -0.08, f'{closest_label}', transform=ax.transAxes, ha='center', va='top', fontsize=12, color='red')

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
        ax2.text(0.5, -0.2, f' {closest_label1}', transform=ax2.transAxes, ha='center', va='top', fontsize=12,
                 color='red')

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

    added_text = f"f{number_f}_chains{num_chains}_k{k}_assumption{assumption_sigma}_noisestd{noise_std}_{prior_var}_k{k}_{max_samples}_{current_time}"

    Xu_pred_mean = jnp.mean(posterior_samples_list, axis=0)

    plot_u_pred(Xu_without_noise, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred_mean, prior_var, assumption_sigma, k,
                max_samples, learning, num_chains, number_f, added_text)
    plot_dist(Xu_without_noise, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred_mean, prior_var, assumption_sigma, k,
              max_samples, learning, num_chains, number_f, posterior_samples_list, prior_samples, number_u, added_text)

