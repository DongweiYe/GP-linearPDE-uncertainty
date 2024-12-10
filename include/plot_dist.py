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
              max_samples, learning, num_chains, number_f, posterior_samples_list, prior_samples, number_u, added_text):
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.figsize': (20, 12),
        'text.usetex': False,
    })
    # num_points = Xu_certain.shape[0]
    num_points = number_u
    fig, axes = plt.subplots(1, num_points, figsize=(20, 5))
    fig.subplots_adjust(top=0.8, wspace=0.4)

    Xu_index = Xu_certain[:4, :]
    u_t_index = jnp.argsort(Xu_index[:, 1])
    # Xu_certain_all = Xu_certain_all[u_t_index]

    for i in range(num_points):
        ax = axes[i]
        idx = u_t_index[i]
        print(f"i: {i}")
        print(f"idx: {idx}")
        print(f"axis: {ax}")
        Xu_index = Xu_certain[:4,:]
        print(f"Xu_certain_all[idx]: {Xu_index[idx, 1]}")

        prior_data = prior_samples[:, idx, :]
        posterior_data = posterior_samples_list[:, idx, :]
        kde_prior = sns.kdeplot(x=prior_data[:, 0], y=prior_data[:, 1], ax=ax, cmap='Oranges', alpha=1, bw_adjust=1.5, zorder=1)
        kde_posterior = sns.kdeplot(x=posterior_data[:, 0], y=posterior_data[:, 1], ax=ax, cmap='Blues',
                                    alpha=1, bw_adjust=1.5, zorder=2)
        # ax.scatter(Xu_certain_all[i, 0], Xu_certain_all[i, 1], color='black', label='GT', marker='o', s=100,linewidths=3)
        # ax.scatter(Xu_noise[i, 0], Xu_noise[i, 1], color='darkorange', label='Xu noise', marker='x', s=100,linewidths=3)
        # ax.scatter(Xu_pred[i, 0], Xu_pred[i, 1], color='darkblue', label='Posterior mean', marker='*', s=100,linewidths=3)

        ax.scatter(Xu_certain[idx, 0], Xu_certain[idx, 1], color='black', edgecolor='black', label='ground truth', marker='o',
                   s=90, linewidths=2, zorder=3)

        ax.scatter(Xu_pred[idx, 0], Xu_pred[idx, 1], color='blue', edgecolor='blue', label='posterior mean', marker='*',
                   s=100, linewidths=2, zorder=5)

        ax.scatter(Xu_noise[idx, 0], Xu_noise[idx, 1], color='darkorange', edgecolor='darkorange', label='prior mean',
                   marker='s', s=60, linewidths=2, zorder=4)

        # posterior_peak_x = find_hist_peak(posterior_data[:, 0])
        # posterior_peak_t = find_hist_peak(posterior_data[:, 1])
        # ax.scatter(posterior_peak_x, posterior_peak_t, color='tab:purple', label='Posterior peak', marker='o')

        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)

        # ax.set_title(f'point {i + 1} with uncertain x and t')

        # ax.set_xlabel('x')
        # ax.set_ylabel('t')
        ax.tick_params(axis='both', which='major')
        # ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    handles.append(plt.Line2D([0], [0], color='orange', lw=4, label='prior distribution'))
    labels.append('prior distribution')
    handles.append(plt.Line2D([0], [0], color='darkblue', lw=4, label='posterior distribution'))
    labels.append('posterior distribution')

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), fontsize=16, ncol=len(labels))

    current_time = datetime.datetime.now().strftime("%M%S")
    fig.savefig(
        f'dist_{added_text}.png')

def plot_dist_rd_2(Xu_certain,
              Xu_noise,
              Xu_pred,
              posterior_samples_list, prior_samples, number_u, added_text, prior_samples_list):
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.figsize': (20, 12),
        'text.usetex': False,
    })
    # num_points = Xu_certain.shape[0]
    num_points = number_u
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    # fig.subplots_adjust(top=0.8, wspace=0.4)

    u_t_index = jnp.argsort(Xu_certain[:, 1])
    for i in range(num_points):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        posterior_data = posterior_samples_list[:, u_t_index[i], :]
        prior_data = prior_samples_list[:, u_t_index[i], :]
        kde_prior = sns.kdeplot(x=prior_data[:, 0], y=prior_data[:, 1], ax=ax, cmap='Oranges', alpha=1, bw_adjust=1.5,
                                zorder=1)
        kde_posterior = sns.kdeplot(x=posterior_data[:, 0], y=posterior_data[:, 1], ax=ax, cmap='Blues', alpha=1,
                                    bw_adjust=1.5, zorder=2)

        ax.scatter(Xu_certain[u_t_index[i], 0], Xu_certain[u_t_index[i], 1], color='black', edgecolor='black',
                   label='ground truth', marker='o', s=110, linewidths=2, zorder=3)
        ax.scatter(Xu_pred[u_t_index[i], 0], Xu_pred[u_t_index[i], 1], color='blue', edgecolor='blue',
                   label='posterior mean', marker='*', s=150, linewidths=2, zorder=5)
        ax.scatter(Xu_noise[u_t_index[i], 0], Xu_noise[u_t_index[i], 1], color='darkorange', edgecolor='darkorange',
                   label='prior mean', marker='s', s=80, linewidths=2, zorder=4)

        ax.tick_params(axis='both', which='major')
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
    handles.append(plt.Line2D([0], [0], color='orange', lw=4, label='prior distribution'))
    labels.append('prior distribution')
    handles.append(plt.Line2D([0], [0], color='darkblue', lw=4, label='posterior distribution'))
    labels.append('posterior distribution')

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), fontsize=16, ncol=len(labels))

    current_time = datetime.datetime.now().strftime("%M%S")
    fig.savefig(
        f'rd_dist_{added_text}.png')

def plot_dist_rd(Xu_certain,
              Xu_noise,
              Xu_pred,
              posterior_samples_list, prior_samples, number_u, added_text, prior_samples_list):
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.figsize': (20, 12),
        'text.usetex': False,
    })
    # num_points = Xu_certain.shape[0]
    num_points = number_u
    fig, axes = plt.subplots(1, num_points, figsize=(20, 5))
    fig.subplots_adjust(top=0.8, wspace=0.4)

    u_t_index = jnp.argsort(Xu_certain[:, 1])
    for i in range(num_points):
        ax = axes[i]

        # prior_data = prior_samples[:, i * 2:(i + 1) * 2]
        posterior_data = posterior_samples_list[:, u_t_index[i], :]
        prior_data = prior_samples_list[:, u_t_index[i], :]
        # kde_posterior = sns.kdeplot(x=posterior_data[:, 0], y=posterior_data[:, 1], ax=ax, color="tab:blue",
        #                             fill=True, alpha=.3, linewidth=0, bw_adjust=2, zorder=2)
        # kde_prior = sns.kdeplot(x=prior_data[:, 0], y=prior_data[:, 1], ax=ax, color = "tab:orange", fill=True, alpha=.3, linewidth=0, bw_adjust=2, zorder=1)
        kde_prior = sns.kdeplot(x=prior_data[:, 0], y=prior_data[:, 1], ax=ax, cmap='Oranges', alpha=1, bw_adjust=1.5,
                                zorder=1)
        kde_posterior = sns.kdeplot(x=posterior_data[:, 0], y=posterior_data[:, 1], ax=ax, cmap='Blues',
                                    alpha=1, bw_adjust=1.5, zorder=2)

        # ax.scatter(Xu_certain_all[i, 0], Xu_certain_all[i, 1], color='black', label='GT', marker='o', s=100,linewidths=3)
        # ax.scatter(Xu_noise[i, 0], Xu_noise[i, 1], color='darkorange', label='Xu noise', marker='x', s=100,linewidths=3)
        # ax.scatter(Xu_pred[i, 0], Xu_pred[i, 1], color='darkblue', label='Posterior mean', marker='*', s=100,linewidths=3)

        ax.scatter(Xu_certain[u_t_index[i], 0], Xu_certain[u_t_index[i], 1], color='black', edgecolor='black', label='ground truth', marker='o',
                   s=110, linewidths=2, zorder=3)

        ax.scatter(Xu_pred[u_t_index[i], 0], Xu_pred[u_t_index[i], 1], color='blue', edgecolor='blue', label='posterior mean', marker='*',
                   s=150, linewidths=2, zorder=5)

        ax.scatter(Xu_noise[u_t_index[i], 0], Xu_noise[u_t_index[i], 1], color='darkorange', edgecolor='darkorange', label='prior mean',
                   marker='s', s=80, linewidths=2, zorder=4)
        # posterior_peak_x = find_hist_peak(posterior_data[:, 0])
        # posterior_peak_t = find_hist_peak(posterior_data[:, 1])
        # ax.scatter(posterior_peak_x, posterior_peak_t, color='tab:purple', label='Posterior peak', marker='o')

        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)

        # ax.set_title(f'point {i + 1} with uncertain x and t')

        # ax.set_xlabel('x')
        # ax.set_ylabel('t')
        ax.tick_params(axis='both', which='major')
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    handles.append(plt.Line2D([0], [0], color='orange', lw=4, label='prior distribution'))
    labels.append('prior distribution')
    handles.append(plt.Line2D([0], [0], color='darkblue', lw=4, label='posterior distribution'))
    labels.append('posterior distribution')

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), fontsize=16, ncol=len(labels))

    current_time = datetime.datetime.now().strftime("%M%S")
    fig.savefig(
        f'rd_dist_{added_text}.png')


def plot_with_noise(number_u, number_u_only_x, posterior_samples_list, prior_samples, Xu_certain, Xu_noise, bw, added_text):
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.figsize': (20, 5),
        'text.usetex': False,
    })

    fig1, axes1 = plt.subplots(1, number_u, figsize=(20, 5))
    fig1.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
    for vague_points in range(number_u):
        ax = axes1[vague_points]
        data = posterior_samples_list[:, vague_points, 0]
        prior_data = prior_samples[:, vague_points, 0]
        ax.axvline(Xu_certain[vague_points, 0], color='black', label='ground truth', linestyle='--', linewidth=2.1)
        #ax.axvline(Xu_noise[vague_points, 0], color='seagreen', label='x noised', linestyle='-', linewidth=2)
        sns.kdeplot(data, ax=ax, color="tab:blue", fill=True, alpha=.3, linewidth=0, label='posterior', bw_adjust=bw)
        sns.kdeplot(prior_data, ax=ax, color='tab:orange', fill=True, alpha=.3, linewidth=0, label='prior')

        posterior_mean = jnp.mean(data)
        ax.axvline(posterior_mean, color="tab:blue", linestyle='--', linewidth=2.2, label='posterior mean')

        # ax.set_xlabel('x')
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
    fig1.savefig(f"kdeplot_x_{added_text}.png", bbox_inches='tight')

    fig2, axes2 = plt.subplots(1, number_u, figsize=(20, 5))
    fig2.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
    for vague_points in range(number_u):
        ax2 = axes2[vague_points]
        data1 = posterior_samples_list[:, vague_points, 1]
        prior_data1 = prior_samples[:, vague_points,  1]
        ax2.axvline(Xu_certain[vague_points, 1], color='black', label='ground truth', linestyle='--', linewidth=2.1)
       # ax2.axvline(Xu_noise[vague_points, 1], color='seagreen', label='t noised', linestyle='-', linewidth=2)
        sns.kdeplot(data1, ax=ax2, color="tab:blue", fill=True, alpha=.3, linewidth=0, label='posterior', bw_adjust=bw)
        sns.kdeplot(prior_data1, ax=ax2, color='tab:orange', fill=True, alpha=.3, linewidth=0, label='prior')

        posterior_mean1 = jnp.mean(data1)
        ax2.axvline(posterior_mean1, color="tab:blue", linestyle='--', linewidth=2.2, label='posterior mean')

        # ax2.set_xlabel('t')
        ax2.tick_params(axis='both', which='major')
        # ax2.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    handles, labels = [], []
    for ax2 in axes2:
        for handle, label in zip(*ax2.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels))

    current_time = datetime.datetime.now().strftime("%M%S")
    fig2.savefig(f"kdeplot_t_{added_text}.png", bbox_inches='tight')

    fig3, axes3 = plt.subplots(1, number_u_only_x, figsize=(20, 5))
    fig3.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
    for vague_points in range(number_u_only_x):
        ax3 = axes3[vague_points]
        data3 = posterior_samples_list[:, (number_u + vague_points), 0]
        prior_data3 = prior_samples[:, (number_u + vague_points), 0]
        ax3.axvline(Xu_certain[number_u + vague_points, 0], color='black', label='ground truth', linestyle='--', linewidth=2.1)
       # ax3.axvline(Xu_noise[number_u + vague_points, 0], color='seagreen', label='x noised', linestyle='-',
       #             linewidth=2)
        sns.kdeplot(data3, ax=ax3, color="tab:blue", label='posterior', bw_adjust=bw,  fill=True, alpha=.3, linewidth=0)
        sns.kdeplot(prior_data3, ax=ax3, color='tab:orange', label='prior', fill=True, alpha=.3, linewidth=0)
        # ax3.axvline(Xu_certain[number_u + vague_points, 0], color='black', label='GT', linestyle='--', linewidth=2.2)

        posterior_mean3 = jnp.mean(data3)
        ax3.axvline(posterior_mean3, color="tab:blue", linestyle='--', linewidth=2.2, label='posterior mean')

        # ax3.set_xlabel('x')
        ax3.tick_params(axis='both', which='major')
        # ax3.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    handles, labels = [], []
    for ax3 in axes3:
        for handle, label in zip(*ax3.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels))

    current_time = datetime.datetime.now().strftime("%M%S")
    fig3.savefig(f"kdeplot_x_only_{added_text}.png", bbox_inches='tight')


def plot_with_noise_rd(number_u, number_u_only_x, posterior_samples_list, prior_samples, Xu_certain, Xu_noise, bw, added_text):
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
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

        ax.axvline(Xu_certain[vague_points, 0], color='black', label='ground truth', linestyle='--', linewidth=2.1)
        #ax.axvline(Xu_noise[vague_points, 0], color='seagreen', label='x noised', linestyle='-', linewidth=2)
        sns.kdeplot(data, ax=ax, color="tab:blue", label='posterior', bw_adjust=bw,  fill=True, alpha=.3, linewidth=0)
        sns.kdeplot(prior_data, ax=ax, color='tab:orange', label='prior',fill=True, alpha=.3, linewidth=0)

        posterior_mean = jnp.mean(data)
        ax.axvline(posterior_mean, color="tab:blue", linestyle='--', linewidth=2.2, label='posterior mean')

        # ax.set_xlabel('x')
        ax.tick_params(axis='both', which='major')
        # ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    handles, labels = [], []
    for ax in axes1:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels))

    current_time = datetime.datetime.now().strftime("%M%S")
    fig1.savefig(f"rd_kdeplot_x_{added_text}.png", bbox_inches='tight')

    fig2, axes2 = plt.subplots(1, number_u, figsize=(20, 5))
    fig2.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
    for vague_points in range(number_u):
        ax2 = axes2[vague_points]
        data1 = posterior_samples_list[:, vague_points, 1]
        prior_data1 = prior_samples[:, vague_points * 2 + 1]
        ax2.axvline(Xu_certain[vague_points, 1],  color='black', label='ground truth', linestyle='--', linewidth=2.1)
       # ax2.axvline(Xu_noise[vague_points, 1], color='seagreen', label='t noised', linestyle='-', linewidth=2)
        sns.kdeplot(data1, ax=ax2,color="tab:blue", fill=True, alpha=.3, linewidth=0, label='posterior', bw_adjust=bw)
        sns.kdeplot(prior_data1, ax=ax2, color='tab:orange', fill=True, alpha=.3, linewidth=0, label='prior')

        posterior_mean1 = jnp.mean(data1)
        ax2.axvline(posterior_mean1, color="tab:blue", linestyle='--', linewidth=2.2, label='posterior mean')

        # ax2.set_xlabel('t')
        ax2.tick_params(axis='both', which='major')
        # ax2.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    handles, labels = [], []
    for ax2 in axes2:
        for handle, label in zip(*ax2.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels))

    current_time = datetime.datetime.now().strftime("%M%S")
    fig2.savefig(f"rd_kdeplot_t_{added_text}.png", bbox_inches='tight')

    # fig3, axes3 = plt.subplots(1, number_u_only_x, figsize=(20, 5))
    # fig3.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
    # for vague_points in range(number_u_only_x):
    #     ax3 = axes3[vague_points]
    #     data3 = posterior_samples_list[:, number_u + vague_points, 0]
    #     prior_data3 = prior_samples[:, (number_u + vague_points) * 2]
    #     ax3.axvline(Xu_certain[number_u + vague_points, 0], color='black', label='x GT', linestyle='-', linewidth=2)
    #    # ax3.axvline(Xu_noise[number_u + vague_points, 0], color='seagreen', label='x noised', linestyle='-',
    #    #             linewidth=2)
    #     sns.kdeplot(data3, ax=ax3, color='darkblue', label='x posterior', bw_adjust=bw, linestyle='--',  alpha=0.6)
    #     sns.kdeplot(prior_data3, ax=ax3, color='tab:orange', label='x prior', linestyle='--',  alpha=0.6)

    #     posterior_mean3 = jnp.mean(data3)
    #     ax3.axvline(posterior_mean3, color='red', linestyle='--', linewidth=2, label='posterior mean')

    #     ax3.set_xlabel('x')
    #     ax3.tick_params(axis='both', which='major')
    #     ax3.grid(True, linestyle='--', linewidth=0.3, alpha=0.3, color='gray')

    # handles, labels = [], []
    # for ax3 in axes3:
    #     for handle, label in zip(*ax3.get_legend_handles_labels()):
    #         if label not in labels:
    #             handles.append(handle)
    #             labels.append(label)
    # fig3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels))

    # current_time = datetime.datetime.now().strftime("%M%S")
    # fig3.savefig(f"kdeplot_x_only_{added_text}.png", bbox_inches='tight')

def plot_and_save_kde_histograms(posterior_samples_list, prior_samples, Xu_certain, Xu_noise, number_u, number_f, num_chains, k, assumption_sigma, prior_std, noise_std, number_init, number_bound, prior_var, max_samples, bw, added_text):
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

        # posterior_peak = find_kde_peak(data)
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

        # posterior_peak1 = find_kde_peak(data1)
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
        f"kdeplot_{added_text}.png",
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
        # posterior_peak = find_kde_peak(posterior_data)
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
        # posterior_peak1 = find_kde_peak(posterior_data1)
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
            f"hist_{added_text}.png",
            bbox_inches='tight')
