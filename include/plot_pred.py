import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import jax.random as random
from mpl_toolkits.axes_grid1 import make_axes_locatable

def prediction_mean(ypred_list):
    return jnp.mean(ypred_list, axis=0)

def prediction_variance(ypred_list, yvar_list):
    ymean_var = jnp.var(ypred_list, axis=0)
    yvar_diag = jnp.diagonal(yvar_list, axis1=1, axis2=2)
    yvar_mean = jnp.mean(yvar_diag, axis=0)

    return ymean_var + yvar_mean

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

def save_individual_plot(data, title, cmap, vmin, vmax, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap(ax, data, title, cmap, vmin, vmax)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_and_save_prediction_results(X_plot_prediction, u_values_gt, y_final_mean_list_prior, y_final_var_list_prior, y_final_mean_list_posterior, y_final_var_list_posterior):
    gp_mean_posterior = prediction_mean(y_final_mean_list_posterior).reshape(X_plot_prediction.shape)
    abs_diff_gt_gp = jnp.abs(u_values_gt - gp_mean_posterior)
    var_prior = prediction_variance(y_final_mean_list_prior, y_final_var_list_prior).reshape(X_plot_prediction.shape)
    var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior).reshape(X_plot_prediction.shape)
    abs_var_diff = jnp.abs(var_prior - var_posterior)

    plot_titles = [
        'Ground Truth',
        'GP Mean Prediction (Posterior)',
        'Absolute Difference (GT vs GP Mean)',
        'Variance with Prior',
        'Variance with Posterior',
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

    current_time = datetime.datetime.now().strftime("%M%S")
    filenames = [
        f"ground_truth_{current_time}.png",
        f"gp_mean_posterior_{current_time}.png",
        f"abs_diff_gt_gp_{current_time}.png",
        f"var_priorv_{current_time}.png",
        f"var_posterior_{current_time}.png",
        f"abs_var_diff_{current_time}.png"
    ]

    for i in range(3):
        data, cmap = plot_data[i]
        save_individual_plot(data, plot_titles[i], cmap, row1_min, row1_max, filenames[i])

    for i in range(3):
        data, cmap = plot_data[i + 3]
        save_individual_plot(data, plot_titles[i + 3], cmap, row2_min, row2_max, filenames[i + 3])

