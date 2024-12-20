import datetime

import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def prediction_mean(ypred_list):
    return jnp.mean(ypred_list, axis=0)


# def prediction_variance(ypred_list, yvar_diag):
#     ymean_var = jnp.var(ypred_list, axis=0)
#
#     yvar_mean = jnp.mean(yvar_diag, axis=0)
#
#     print("Check ypred_list NaN:", jnp.isnan(ypred_list).any(), "Inf:", jnp.isinf(ypred_list).any())
#     print("Check yvar_diag NaN:", jnp.isnan(yvar_diag).any(), "Inf:", jnp.isinf(yvar_diag).any())
#
#     return ymean_var + yvar_mean

def prediction_variance(ypred_list, yvar_diag):
    if jnp.isnan(ypred_list).any() or jnp.isinf(ypred_list).any():
        print("ypred_list contains NaN or Inf")
    if jnp.isnan(yvar_diag).any() or jnp.isinf(yvar_diag).any():
        print("yvar_diag contains NaN or Inf")

    ymean_var = jnp.var(ypred_list, axis=0)
    yvar_mean = jnp.mean(yvar_diag, axis=0)

    if jnp.isnan(ymean_var).any() or jnp.isinf(ymean_var).any():
        print("ymean_var is NaN or Inf")
    if jnp.isnan(yvar_mean).any() or jnp.isinf(yvar_mean).any():
        print("yvar_mean is NaN or Inf")

    result = ymean_var + yvar_mean
    if jnp.isnan(result).any() or jnp.isinf(result).any():
        print("result is NaN or Inf")

    return result


def plot_heatmap(ax, data, title, cmap, vmin, vmax):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])

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
        'Ground Truth',
        'GP Prediction Mean (prior)',
        'Absolute Error (GT vs prior)',
        'Ground Truth',
        'GP Prediction Mean (posterior)',
        'Absolute Error (GT vs posterior)',
        'GP variance (prior)',
        'GP variance (posterior)',
        'Absolute Difference'
    ]
    cmap1 = 'viridis'
    cmap2 = 'plasma_r'
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

    row1_min = jnp.min(plot_data[0][0])
    row1_max = jnp.max(plot_data[0][0])

    row2_min = jnp.min(plot_data[3][0])
    row2_max = jnp.max(plot_data[3][0])

    row3_min = min(jnp.min(plot_data[6][0]), jnp.min(plot_data[7][0]))
    row3_max = max(jnp.max(plot_data[6][0]), jnp.max(plot_data[7][0]))

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
        if i < 2:
            vmin, vmax = row1_min, row1_max
        elif i < 3:
            vmin, vmax = jnp.min(plot_data[2][0]), jnp.max(plot_data[2][0])
        elif i < 5:
            vmin, vmax = row2_min, row2_max
        elif i < 6:
            vmin, vmax = jnp.min(plot_data[5][0]), jnp.max(plot_data[5][0])
        elif i < 8:
            vmin, vmax = row3_min, row3_max
        else:
            vmin, vmax = jnp.min(plot_data[8][0]), jnp.max(plot_data[8][0])
        save_individual_plot(data, plot_titles[i], cmap, vmin, vmax, filenames[i])


def plot_heatmap_rd(ax, data, title, cmap, vmin, vmax):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=[-1, 1, 1, 0])

    ax.tick_params(axis='both', which='major', labelsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14)


def save_individual_plot_rd(data, title, cmap, vmin, vmax, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap_rd(ax, data, title, cmap, vmin, vmax)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_and_save_prediction_results_rd(u_values_gt,
                                        gp_mean_prior,
                                        abs_diff_prior,
                                        gp_mean_posterior,
                                        abs_diff_gt_gp,
                                        var_prior,
                                        var_posterior,
                                        abs_var_diff, added_text):
    plot_titles = [
        'Ground Truth',
        'GP Prediction Mean (prior)',
        'Absolute Error (GT vs prior)',
        'Ground Truth',
        'GP Prediction Mean (posterior)',
        'Absolute Error (GT vs posterior)',
        'GP variance (prior)',
        'GP variance (posterior)',
        'Absolute Difference'
    ]
    cmap1 = 'viridis'
    cmap2 = 'plasma_r'
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

    row1_min = jnp.min(plot_data[0][0])
    row1_max = jnp.max(plot_data[0][0])

    row2_min = jnp.min(plot_data[3][0])
    row2_max = jnp.max(plot_data[3][0])

    row3_min = min(jnp.min(plot_data[6][0]), jnp.min(plot_data[7][0]))
    row3_max = max(jnp.max(plot_data[6][0]), jnp.max(plot_data[7][0]))

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
        if i < 2:
            vmin, vmax = row1_min, row1_max
        elif i < 3:
            vmin, vmax = jnp.min(plot_data[2][0]), jnp.max(plot_data[2][0])
        elif i < 5:
            vmin, vmax = row2_min, row2_max
        elif i < 6:
            vmin, vmax = jnp.min(plot_data[5][0]), jnp.max(plot_data[5][0])
        elif i < 8:
            vmin, vmax = row3_min, row3_max
        else:
            vmin, vmax = jnp.min(plot_data[8][0]), jnp.max(plot_data[8][0])
        save_individual_plot_rd(data, plot_titles[i], cmap, vmin, vmax, filenames[i])


def plot_and_save_prediction_results_combine(u_values_gt,
                                             gp_mean_prior,
                                             abs_diff_prior,
                                             gp_mean_posterior,
                                             abs_diff_gt_gp,
                                             var_prior,
                                             var_posterior,
                                             abs_var_diff, added_text):
    plot_titles = [
        'Ground Truth',
        'GP Prediction Mean (prior)',
        'Absolute Error (GT vs prior)',
        'Ground Truth',
        'GP Prediction Mean (posterior)',
        'Absolute Error (GT vs posterior)',
        'GP variance (prior)',
        'GP variance (posterior)',
        'Absolute Difference'
    ]
    cmap1 = 'viridis'
    cmap2 = 'plasma_r'
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

    row1_min = jnp.min(plot_data[0][0])
    row1_max = jnp.max(plot_data[0][0])

    row2_min = jnp.min(plot_data[3][0])
    row2_max = jnp.max(plot_data[3][0])

    row3_min = min(jnp.min(plot_data[6][0]), jnp.min(plot_data[7][0]))
    row3_max = max(jnp.max(plot_data[6][0]), jnp.max(plot_data[7][0]))

    title_size = 20
    label_size = 16
    cbar_size = 14
    fig1, axs1 = plt.subplots(2, 3, figsize=(20, 10))
    for i in range(2):
        data, cmap = plot_data[i]
        im = axs1[0, i].imshow(data, cmap=cmap, vmin=row1_min, vmax=row1_max, extent=[0, 1, 0, 1])

        axs1[0, i].tick_params(axis='both', labelsize=label_size)
        fig1.colorbar(im, ax=axs1[0, i]).ax.tick_params(labelsize=cbar_size)

    data2, cmap2 = plot_data[2]
    im2 = axs1[0, 2].imshow(data2, cmap=cmap2, extent=[0, 1, 0, 1])

    axs1[0, 2].tick_params(axis='both', labelsize=label_size)
    fig1.colorbar(im2, ax=axs1[0, 2]).ax.tick_params(labelsize=cbar_size)

    for i in range(2):
        data, cmap = plot_data[i + 3]
        im = axs1[1, i].imshow(data, cmap=cmap, vmin=row2_min, vmax=row2_max, extent=[0, 1, 0, 1])

        axs1[1, i].tick_params(axis='both', labelsize=label_size)
        fig1.colorbar(im, ax=axs1[1, i]).ax.tick_params(labelsize=cbar_size)

    data5, cmap5 = plot_data[5]
    im5 = axs1[1, 2].imshow(data5, cmap=cmap5, extent=[0, 1, 0, 1])

    axs1[1, 2].tick_params(axis='both', labelsize=label_size)
    fig1.colorbar(im5, ax=axs1[1, 2]).ax.tick_params(labelsize=cbar_size)

    plt.tight_layout()
    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f"part1_{added_text}_{current_time}.png")

    fig2, axs2 = plt.subplots(1, 3, figsize=(20, 5))

    for i in range(2):
        data, cmap = plot_data[i + 6]
        im = axs2[i].imshow(data, cmap=cmap, vmin=row3_min, vmax=row3_max, extent=[0, 1, 0, 1])

        axs2[i].tick_params(axis='both', labelsize=label_size)
        fig2.colorbar(im, ax=axs2[i]).ax.tick_params(labelsize=cbar_size)

    data8, cmap8 = plot_data[8]
    im8 = axs2[2].imshow(data8, cmap=cmap8, extent=[0, 1, 0, 1])

    axs2[2].tick_params(axis='both', labelsize=label_size)
    fig2.colorbar(im8, ax=axs2[2]).ax.tick_params(labelsize=cbar_size)

    plt.tight_layout()
    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f"part2_{added_text}_{current_time}.png")


def plot_and_save_prediction_results_combine_rd(u_values_gt,
                                                gp_mean_prior,
                                                abs_diff_prior,
                                                gp_mean_posterior,
                                                abs_diff_gt_gp,
                                                var_prior,
                                                var_posterior,
                                                abs_var_diff, added_text):
    plot_titles = [
        'Ground Truth',
        'GP Prediction Mean (prior)',
        'Absolute Error (GT vs prior)',
        'Ground Truth',
        'GP Prediction Mean (posterior)',
        'Absolute Error (GT vs posterior)',
        'GP variance (prior)',
        'GP variance (posterior)',
        'Absolute Difference'
    ]
    cmap1 = 'viridis'
    cmap2 = 'plasma_r'
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

    row1_min = jnp.min(plot_data[0][0])
    row1_max = jnp.max(plot_data[0][0])

    row2_min = jnp.min(plot_data[3][0])
    row2_max = jnp.max(plot_data[3][0])

    row3_min = min(jnp.min(plot_data[6][0]), jnp.min(plot_data[7][0]))
    row3_max = max(jnp.max(plot_data[6][0]), jnp.max(plot_data[7][0]))

    title_size = 20
    label_size = 16
    cbar_size = 14
    fig1, axs1 = plt.subplots(2, 3, figsize=(20, 6))
    for i in range(2):
        data, cmap = plot_data[i]
        im = axs1[0, i].imshow(data, cmap=cmap, vmin=row1_min, vmax=row1_max, extent=[-1, 1, 1, 0])

        axs1[0, i].tick_params(axis='both', labelsize=label_size)
        fig1.colorbar(im, ax=axs1[0, i]).ax.tick_params(labelsize=cbar_size)

    data2, cmap2 = plot_data[2]
    im2 = axs1[0, 2].imshow(data2, cmap=cmap2, extent=[-1, 1, 1, 0])

    axs1[0, 2].tick_params(axis='both', labelsize=label_size)
    fig1.colorbar(im2, ax=axs1[0, 2]).ax.tick_params(labelsize=cbar_size)

    for i in range(2):
        data, cmap = plot_data[i + 3]
        im = axs1[1, i].imshow(data, cmap=cmap, vmin=row2_min, vmax=row2_max, extent=[-1, 1, 1, 0])

        axs1[1, i].tick_params(axis='both', labelsize=label_size)
        fig1.colorbar(im, ax=axs1[1, i]).ax.tick_params(labelsize=cbar_size)

    data5, cmap5 = plot_data[5]
    im5 = axs1[1, 2].imshow(data5, cmap=cmap5, extent=[-1, 1, 1, 0])

    axs1[1, 2].tick_params(axis='both', labelsize=label_size)
    fig1.colorbar(im5, ax=axs1[1, 2]).ax.tick_params(labelsize=cbar_size)

    plt.tight_layout()
    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f"part1_{added_text}_{current_time}.png")

    fig2, axs2 = plt.subplots(1, 3, figsize=(20, 3))

    for i in range(2):
        data, cmap = plot_data[i + 6]
        im = axs2[i].imshow(data, cmap=cmap, vmin=row3_min, vmax=row3_max, extent=[-1, 1, 1, 0])

        axs2[i].tick_params(axis='both', labelsize=label_size)
        fig2.colorbar(im, ax=axs2[i]).ax.tick_params(labelsize=cbar_size)

    data8, cmap8 = plot_data[8]
    im8 = axs2[2].imshow(data8, cmap=cmap8, extent=[-1, 1, 1, 0])

    axs2[2].tick_params(axis='both', labelsize=label_size)
    fig2.colorbar(im8, ax=axs2[2]).ax.tick_params(labelsize=cbar_size)

    plt.tight_layout()
    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f"part2_{added_text}_{current_time}.png")
