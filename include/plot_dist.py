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
                max_samples, learning, num_chains, number_f, posterior_samples_list, prior_samples):
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
    num_points = Xu_certain.shape[0]
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
