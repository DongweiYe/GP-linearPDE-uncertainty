import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.stats import gaussian_kde


def find_hist_peak(data, bins=30):
    hist, bin_edges = jnp.histogram(data, bins=bins)
    peak_index = jnp.argmax(hist)
    peak = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2
    return peak


def plot_dist(Xu_certain_all, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred, prior_var, assumption_sigma, k,
                max_samples, learning, num_chains, number_f, posterior_samples_list, prior_samples,number_u):
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'figure.figsize': (20, 12),
        'text.usetex': False,
    })
    # num_points = Xu_certain.shape[0]
    num_points = number_u
    fig, axes = plt.subplots(1, num_points, figsize=(20, 5))
    fig.subplots_adjust(top=0.8, wspace=0.4)

    for i in range(num_points):
        ax = axes[i]

        prior_data = prior_samples[:, i * 2:(i + 1) * 2]
        posterior_data = posterior_samples_list[:, i, :]
        kde_prior = sns.kdeplot(x=prior_data[:, 0], y=prior_data[:, 1], ax=ax, fill=True, cmap='Oranges', alpha=0.6)
        kde_posterior = sns.kdeplot(x=posterior_data[:, 0], y=posterior_data[:, 1], ax=ax, fill=True, cmap='Blues',
                                    alpha=0.6)

        ax.scatter(Xu_certain_all[i, 0], Xu_certain_all[i, 1], color='black', label='GT', marker='o')
        ax.scatter(Xu_noise[i, 0], Xu_noise[i, 1], color='tab:green', label='Xu noise', marker='x')
        ax.scatter(Xu_pred[i, 0], Xu_pred[i, 1], color='tab:red', label='Posterior mean', marker='o')

        # posterior_peak_x = find_hist_peak(posterior_data[:, 0])
        # posterior_peak_t = find_hist_peak(posterior_data[:, 1])
       # ax.scatter(posterior_peak_x, posterior_peak_t, color='tab:purple', label='Posterior peak', marker='o')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_title(f'point {i + 1} with uncertain x and t')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.tick_params(axis='both', which='major')
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    handles.append(plt.Line2D([0], [0], color='orange', lw=4, label='Prior distribution'))
    labels.append('Prior distribution')
    handles.append(plt.Line2D([0], [0], color='darkblue', lw=4, label='Posterior distribution'))
    labels.append('Posterior distribution')

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), fontsize=16, ncol=len(labels))

    current_time = datetime.datetime.now().strftime("%M%S")
    fig.savefig(
        f'dist_f{number_f}_chains{num_chains}_learning{learning}_k{k}_priorvar_{prior_var}_assump{assumption_sigma}_nstd{noise_std}_iter{max_samples}_{current_time}.png')


def plot_with_noise(number_u, number_u_only_x, posterior_samples_list, prior_samples, Xu_certain, Xu_noise, bw):
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'figure.figsize': (20, 5),
        'text.usetex': False,
    })

    fig1, axes1 = plt.subplots(1, number_u, figsize=(20, 5))
    fig1.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
    for vague_points in range(number_u):
        ax = axes1[vague_points]
        data = posterior_samples_list[:, vague_points, 0]
        prior_data = prior_samples[:, vague_points * 2]
        ax.axvline(Xu_certain[vague_points, 0], color='black', label='x GT', linestyle='-', linewidth=2)
        ax.axvline(Xu_noise[vague_points, 0], color='seagreen', label='x noised', linestyle='-', linewidth=2)
        sns.kdeplot(data, ax=ax, color='darkblue', label='x posterior', bw_adjust=bw, linestyle='--',  alpha=0.6)
        sns.kdeplot(prior_data, ax=ax, color='tab:orange', label='x prior', linestyle='--',  alpha=0.6)

        posterior_mean = jnp.mean(data)
        ax.axvline(posterior_mean, color='red', linestyle='--', linewidth=2, label='posterior mean')

        ax.set_xlabel(f'x_uncertain{vague_points}')
        ax.tick_params(axis='both', which='major')
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    handles, labels = [], []
    for ax in axes1:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels))

    current_time = datetime.datetime.now().strftime("%M%S")
    fig1.savefig(f"kdeplot_x_{current_time}.png", bbox_inches='tight')

    fig2, axes2 = plt.subplots(1, number_u, figsize=(20, 5))
    fig2.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
    for vague_points in range(number_u):
        ax2 = axes2[vague_points]
        data1 = posterior_samples_list[:, vague_points, 1]
        prior_data1 = prior_samples[:, vague_points * 2 + 1]
        ax2.axvline(Xu_certain[vague_points, 1], color='black', label='t GT', linestyle='-', linewidth=2)
        ax2.axvline(Xu_noise[vague_points, 1], color='seagreen', label='t noised', linestyle='-', linewidth=2)
        sns.kdeplot(data1, ax=ax2, color='darkblue', label='t posterior', bw_adjust=bw, linestyle='--',  alpha=0.6)
        sns.kdeplot(prior_data1, ax=ax2, color='tab:orange', label='t prior', linestyle='--',  alpha=0.6)

        posterior_mean1 = jnp.mean(data1)
        ax2.axvline(posterior_mean1, color='red', linestyle='--', linewidth=2, label='posterior mean')

        ax2.set_xlabel(f't_uncertain{vague_points}')
        ax2.tick_params(axis='both', which='major')
        ax2.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    handles, labels = [], []
    for ax2 in axes2:
        for handle, label in zip(*ax2.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels))

    current_time = datetime.datetime.now().strftime("%M%S")
    fig2.savefig(f"kdeplot_t_{current_time}.png", bbox_inches='tight')

    fig3, axes3 = plt.subplots(1, number_u_only_x, figsize=(20, 5))
    fig3.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
    for vague_points in range(number_u_only_x):
        ax3 = axes3[vague_points]
        data3 = posterior_samples_list[:, number_u + vague_points, 0]
        prior_data3 = prior_samples[:, (number_u + vague_points) * 2]
        ax3.axvline(Xu_certain[number_u + vague_points, 0], color='black', label='x GT', linestyle='-', linewidth=2)
        ax3.axvline(Xu_noise[number_u + vague_points, 0], color='seagreen', label='x noised', linestyle='-',
                    linewidth=2)
        sns.kdeplot(data3, ax=ax3, color='darkblue', label='x posterior', bw_adjust=bw, linestyle='--',  alpha=0.6)
        sns.kdeplot(prior_data3, ax=ax3, color='tab:orange', label='x prior', linestyle='--',  alpha=0.6)

        posterior_mean3 = jnp.mean(data3)
        ax3.axvline(posterior_mean3, color='red', linestyle='--', linewidth=2, label='posterior mean')

        ax3.set_xlabel(f'Xu_only_x_uncertain_{vague_points}')
        ax3.tick_params(axis='both', which='major')
        ax3.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    handles, labels = [], []
    for ax3 in axes3:
        for handle, label in zip(*ax3.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels))

    current_time = datetime.datetime.now().strftime("%M%S")
    fig3.savefig(f"kdeplot_x_only_{current_time}.png", bbox_inches='tight')