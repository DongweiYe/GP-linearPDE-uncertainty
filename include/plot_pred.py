import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import jax.random as random
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime

def prediction_mean(ypred_list):
    return jnp.mean(ypred_list, axis=0)

def prediction_variance(ypred_list, yvar_diag):
    ymean_var = jnp.var(ypred_list, axis=0)
    # yvar_diag = jnp.diagonal(yvar_list, axis1=1, axis2=2)
    yvar_mean = jnp.mean(yvar_diag, axis=0)

    return ymean_var + yvar_mean

def plot_heatmap(ax, data, title, cmap, vmin, vmax):
    im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('t', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14)

def save_individual_plot(data, title, cmap, vmin, vmax, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap(ax, data, title, cmap, vmin, vmax)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_and_save_prediction_results(u_values_gt,
                                     gp_mean_prior,
                                     abs_diff_prior,
                                    gp_mean_posterior,
                                    abs_diff_gt_gp,
                                    var_prior,
                                    var_posterior,
                                    abs_var_diff, added_text):

    plot_titles = [
        'Ground truth',
        'GP prediction mean (prior)',
        'Absolute error (prior)',
        'Ground truth',
        'GP prediction mean (posterior)',
        'Absolute error (posterior)',
        'GP variance (prior)',
        'GP variance (posterior)',
        'Absolute variance difference'
    ]
    cmap1 = 'plasma'
    cmap2 = 'inferno'
    plot_data = [
        (u_values_gt, cmap1),
        (gp_mean_prior, cmap1),
        (abs_diff_prior, cmap1),
        (u_values_gt, cmap1),
        (gp_mean_posterior, cmap1),
        (abs_diff_gt_gp, cmap1),
        (var_prior, cmap2),
        (var_posterior, cmap2),
        (abs_var_diff, cmap2)
    ]

    row1_min = min(jnp.min(plot_data[0][0]), jnp.min(plot_data[1][0]), jnp.min(plot_data[2][0]))
    row1_max = max(jnp.max(plot_data[0][0]), jnp.max(plot_data[1][0]), jnp.max(plot_data[2][0]))

    row2_min = min(jnp.min(plot_data[3][0]), jnp.min(plot_data[4][0]), jnp.min(plot_data[5][0]))
    row2_max = max(jnp.max(plot_data[3][0]), jnp.max(plot_data[4][0]), jnp.max(plot_data[5][0]))

    row3_min = min(jnp.min(plot_data[6][0]), jnp.min(plot_data[7][0]), jnp.min(plot_data[8][0]))
    row3_max = max(jnp.max(plot_data[6][0]), jnp.max(plot_data[7][0]), jnp.max(plot_data[8][0]))

    current_time = datetime.datetime.now().strftime("%M%S")
    filenames = [
        f"ground_truth_{added_text}_{current_time}.png",
        f"gp_mean_prior_{added_text}_{current_time}.png",
        f"abs_diff_prior_{added_text}_{current_time}.png",
        f"ground_truth_posterior_{added_text}_{current_time}.png",
        f"gp_mean_posterior_{added_text}_{current_time}.png",
        f"abs_diff_posterior_{added_text}_{current_time}.png",
        f"var_prior_{added_text}_{current_time}.png",
        f"var_posterior_{added_text}_{current_time}.png",
        f"abs_var_diff_{added_text}_{current_time}.png"
    ]

    for i in range(9):
        data, cmap = plot_data[i]
        if i < 3:
            vmin, vmax = row1_min, row1_max
        elif i < 6:
            vmin, vmax = row2_min, row2_max
        else:
            vmin, vmax = row3_min, row3_max
        save_individual_plot(data, plot_titles[i], cmap, vmin, vmax, filenames[i])


def plot_and_save_prediction_results_combine(u_values_gt,
                                     gp_mean_posterior,
                                     abs_diff_gt_gp,
                                     var_prior,
                                     var_posterior,
                                     abs_var_diff, added_text):

    plot_titles = [
        'Ground Truth',
        'GP Mean Prediction (posterior)',
        'Absolute Difference (GT vs GP Mean)',
        'GP variance (prior)',
        'GP variance (posterior)',
        'Absolute Variance Difference'
    ]
    cmap1 = 'GnBu'
    cmap2 = 'PuRd'
    plot_data = [
        (u_values_gt, cmap1),
        (gp_mean_posterior, cmap1),
        (abs_diff_gt_gp, cmap1),
        (var_prior, cmap2),
        (var_posterior, cmap2),
        (abs_var_diff, cmap2)
    ]

    row1_min = min(jnp.min(plot_data[0][0]), jnp.min(plot_data[1][0]), jnp.min(plot_data[2][0]))
    row1_max = max(jnp.max(plot_data[0][0]), jnp.max(plot_data[1][0]), jnp.max(plot_data[2][0]))

    row2_min = min(jnp.min(plot_data[3][0]), jnp.min(plot_data[4][0]), jnp.min(plot_data[5][0]))
    row2_max = max(jnp.max(plot_data[3][0]), jnp.max(plot_data[4][0]), jnp.max(plot_data[5][0]))

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    for i in range(3):
        data, cmap = plot_data[i]
        im = axs[0, i].imshow(data, cmap=cmap, vmin=row1_min, vmax=row1_max)
        axs[0, i].set_title(plot_titles[i])
        fig.colorbar(im, ax=axs[0, i])

    for i in range(3):
        data, cmap = plot_data[i + 3]
        im = axs[1, i].imshow(data, cmap=cmap, vmin=row2_min, vmax=row2_max)
        axs[1, i].set_title(plot_titles[i + 3])
        fig.colorbar(im, ax=axs[1, i])

    plt.tight_layout()
    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f"combined_plot_{added_text}_{current_time}.png")
    plt.show()

