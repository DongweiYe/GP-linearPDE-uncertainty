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
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
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
    cmap1 = 'PuBuGn'
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

    # row1_min = min(jnp.min(plot_data[0][0]), jnp.min(plot_data[1][0]), jnp.min(plot_data[2][0]))
    # row1_max = max(jnp.max(plot_data[0][0]), jnp.max(plot_data[1][0]), jnp.max(plot_data[2][0]))
    #
    # row2_min = min(jnp.min(plot_data[3][0]), jnp.min(plot_data[4][0]), jnp.min(plot_data[5][0]))
    # row2_max = max(jnp.max(plot_data[3][0]), jnp.max(plot_data[4][0]), jnp.max(plot_data[5][0]))
    #
    # row3_min = min(jnp.min(plot_data[6][0]), jnp.min(plot_data[7][0]), jnp.min(plot_data[8][0]))
    # row3_max = max(jnp.max(plot_data[6][0]), jnp.max(plot_data[7][0]), jnp.max(plot_data[8][0]))
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
        elif i<3:
            vmin, vmax = jnp.min(plot_data[2][0]), jnp.max(plot_data[2][0])
        elif i < 5:
            vmin, vmax = row2_min, row2_max
        elif i<6:
            vmin, vmax = jnp.min(plot_data[5][0]), jnp.max(plot_data[5][0])
        elif i < 8:
            vmin, vmax = row3_min, row3_max
        else:
            vmin, vmax = jnp.min(plot_data[8][0]), jnp.max(plot_data[8][0])
        save_individual_plot(data, plot_titles[i], cmap, vmin, vmax, filenames[i])


def plot_heatmap_rd(ax, data, title, cmap, vmin, vmax):
    im = ax.imshow(data, cmap=cmap,  vmin=vmin, vmax=vmax, extent=[-1,1,1,0])
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('t', fontsize=18)
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
    cmap1 = 'PuBuGn'
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

    # row1_min = min(jnp.min(plot_data[0][0]), jnp.min(plot_data[1][0]), jnp.min(plot_data[2][0]))
    # row1_max = max(jnp.max(plot_data[0][0]), jnp.max(plot_data[1][0]), jnp.max(plot_data[2][0]))
    #
    # row2_min = min(jnp.min(plot_data[3][0]), jnp.min(plot_data[4][0]), jnp.min(plot_data[5][0]))
    # row2_max = max(jnp.max(plot_data[3][0]), jnp.max(plot_data[4][0]), jnp.max(plot_data[5][0]))
    #
    # row3_min = min(jnp.min(plot_data[6][0]), jnp.min(plot_data[7][0]), jnp.min(plot_data[8][0]))
    # row3_max = max(jnp.max(plot_data[6][0]), jnp.max(plot_data[7][0]), jnp.max(plot_data[8][0]))
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
        elif i<3:
            vmin, vmax = jnp.min(plot_data[2][0]), jnp.max(plot_data[2][0])
        elif i < 5:
            vmin, vmax = row2_min, row2_max
        elif i<6:
            vmin, vmax = jnp.min(plot_data[5][0]), jnp.max(plot_data[5][0])
        elif i < 8:
            vmin, vmax = row3_min, row3_max
        else:
            vmin, vmax = jnp.min(plot_data[8][0]), jnp.max(plot_data[8][0])
        save_individual_plot_rd(data, plot_titles[i], cmap, vmin, vmax, filenames[i])

# def plot_and_save_prediction_results_combine_6(u_values_gt,
#                                      gp_mean_posterior,
#                                      abs_diff_gt_gp,
#                                      var_prior,
#                                      var_posterior,
#                                      abs_var_diff, added_text):
#
#     plot_titles = [
#         'Ground Truth',
#         'GP Mean Prediction (posterior)',
#         'Absolute Difference (GT vs GP Mean)',
#         'GP variance (prior)',
#         'GP variance (posterior)',
#         'Absolute Variance Difference'
#     ]
#     cmap1 = 'GnBu'
#     cmap2 = 'PuRd'
#     plot_data = [
#         (u_values_gt, cmap1),
#         (gp_mean_posterior, cmap1),
#         (abs_diff_gt_gp, cmap1),
#         (var_prior, cmap2),
#         (var_posterior, cmap2),
#         (abs_var_diff, cmap2)
#     ]
#
#     row1_min = min(jnp.min(plot_data[0][0]), jnp.min(plot_data[1][0]), jnp.min(plot_data[2][0]))
#     row1_max = max(jnp.max(plot_data[0][0]), jnp.max(plot_data[1][0]), jnp.max(plot_data[2][0]))
#
#     row2_min = min(jnp.min(plot_data[3][0]), jnp.min(plot_data[4][0]), jnp.min(plot_data[5][0]))
#     row2_max = max(jnp.max(plot_data[3][0]), jnp.max(plot_data[4][0]), jnp.max(plot_data[5][0]))
#
#     fig, axs = plt.subplots(2, 3, figsize=(18, 10))
#
#     for i in range(3):
#         data, cmap = plot_data[i]
#         im = axs[0, i].imshow(data, cmap=cmap, vmin=row1_min, vmax=row1_max)
#         axs[0, i].set_title(plot_titles[i])
#         fig.colorbar(im, ax=axs[0, i])
#
#     for i in range(3):
#         data, cmap = plot_data[i + 3]
#         im = axs[1, i].imshow(data, cmap=cmap, vmin=row2_min, vmax=row2_max)
#         axs[1, i].set_title(plot_titles[i + 3])
#         fig.colorbar(im, ax=axs[1, i])
#
#     plt.tight_layout()
#     current_time = datetime.datetime.now().strftime("%M%S")
#     plt.savefig(f"combined_plot_{added_text}_{current_time}.png")
#     plt.show()


# def plot_and_save_prediction_results_combine(u_values_gt,
#                                      gp_mean_prior,
#                                      abs_diff_prior,
#                                      gp_mean_posterior,
#                                      abs_diff_gt_gp,
#                                      var_prior,
#                                      var_posterior,
#                                      abs_var_diff, added_text):
#
#     plot_titles = [
#         'Ground Truth',
#         'GP Mean Prediction (prior)',
#         'Absolute Difference (GT vs prior)',
#         'Ground Truth',
#         'GP Mean Prediction (posterior)',
#         'Absolute Difference (GT vs posterior)',
#         'GP variance (prior)',
#         'GP variance (posterior)',
#         'Absolute Variance Difference'
#     ]
#     cmap1 = 'GnBu'
#     cmap2 = 'PuRd'
#     plot_data = [
#         (u_values_gt, cmap1),
#         (gp_mean_prior, cmap1),
#         (abs_diff_prior, cmap1),
#         (u_values_gt, cmap1),
#         (gp_mean_posterior, cmap1),
#         (abs_diff_gt_gp, cmap1),
#         (var_prior, cmap2),
#         (var_posterior, cmap2),
#         (abs_var_diff, cmap2)
#     ]
#
#     row1_min = min(jnp.min(plot_data[0][0]), jnp.min(plot_data[1][0]))
#     row1_max = max(jnp.max(plot_data[0][0]), jnp.max(plot_data[1][0]))
#
#     row2_min = min(jnp.min(plot_data[3][0]), jnp.min(plot_data[4][0]))
#     row2_max = max(jnp.max(plot_data[3][0]), jnp.max(plot_data[4][0]))
#
#     row3_min = min(jnp.min(plot_data[6][0]), jnp.min(plot_data[7][0]))
#     row3_max = max(jnp.max(plot_data[6][0]), jnp.max(plot_data[7][0]))
#
#     fig, axs = plt.subplots(3, 3, figsize=(20, 16))
#
#     for i in range(2):
#         data, cmap = plot_data[i]
#         im = axs[0, i].imshow(data, cmap=cmap, vmin=row1_min, vmax=row1_max)
#         axs[0, i].set_title(plot_titles[i])
#         fig.colorbar(im, ax=axs[0, i])
#
#     data2, cmap2 = plot_data[2]
#     im2 = axs[0, 2].imshow(data2, cmap=cmap2)
#     axs[0, 2].set_title(plot_titles[2])
#     fig.colorbar(im2, ax=axs[0, 2])
#
#     for i in range(2):
#         data, cmap = plot_data[i + 3]
#         im = axs[1, i].imshow(data, cmap=cmap, vmin=row2_min, vmax=row2_max)
#         axs[1, i].set_title(plot_titles[i + 3])
#         fig.colorbar(im, ax=axs[1, i])
#
#     data5, cmap5 = plot_data[5]
#     im5 = axs[1, 2].imshow(data5, cmap=cmap5)
#     axs[1, 2].set_title(plot_titles[5])
#     fig.colorbar(im5, ax=axs[1, 2])
#
#     for i in range(2):
#         data, cmap = plot_data[i + 6]
#         im = axs[2, i].imshow(data, cmap=cmap, vmin=row3_min, vmax=row3_max)
#         axs[2, i].set_title(plot_titles[i + 6])
#         fig.colorbar(im, ax=axs[2, i])
#
#     data8, cmap8 = plot_data[8]
#     im8 = axs[2, 2].imshow(data8, cmap=cmap8)
#     axs[2, 2].set_title(plot_titles[8])
#     fig.colorbar(im8, ax=axs[2, 2])
#
#     plt.tight_layout()
#     current_time = datetime.datetime.now().strftime("%M%S")
#     plt.savefig(f"combined_plot_{added_text}_{current_time}.png")


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
    cmap1 = 'PuBuGn'
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
        im = axs1[0, i].imshow(data, cmap=cmap, vmin=row1_min, vmax=row1_max, extent = [0, 1, 0, 1])
        axs1[0, i].set_title(plot_titles[i], fontsize=title_size)
        axs1[0, i].tick_params(axis='both', labelsize=label_size)
        fig1.colorbar(im, ax=axs1[0, i]).ax.tick_params(labelsize=cbar_size)

    data2, cmap2 = plot_data[2]
    im2 = axs1[0, 2].imshow(data2, cmap=cmap2, extent = [0, 1, 0, 1])
    axs1[0, 2].set_title(plot_titles[2], fontsize=title_size)
    axs1[0, 2].tick_params(axis='both', labelsize=label_size)
    fig1.colorbar(im2, ax=axs1[0, 2]).ax.tick_params(labelsize=cbar_size)

    for i in range(2):
        data, cmap = plot_data[i + 3]
        im = axs1[1, i].imshow(data, cmap=cmap, vmin=row2_min, vmax=row2_max, extent = [0, 1, 0, 1])
        axs1[1, i].set_title(plot_titles[i + 3], fontsize=title_size)
        axs1[1, i].tick_params(axis='both', labelsize=label_size)
        fig1.colorbar(im, ax=axs1[1, i]).ax.tick_params(labelsize=cbar_size)

    data5, cmap5 = plot_data[5]
    im5 = axs1[1, 2].imshow(data5, cmap=cmap5, extent = [0, 1, 0, 1])
    axs1[1, 2].set_title(plot_titles[5], fontsize=title_size)
    axs1[1, 2].tick_params(axis='both', labelsize=label_size)
    fig1.colorbar(im5, ax=axs1[1, 2]).ax.tick_params(labelsize=cbar_size)

    plt.tight_layout()
    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f"part1_{added_text}_{current_time}.png")

    fig2, axs2 = plt.subplots(1, 3, figsize=(20, 5))

    for i in range(2):
        data, cmap = plot_data[i + 6]
        im = axs2[i].imshow(data, cmap=cmap, vmin=row3_min, vmax=row3_max, extent = [0, 1, 0, 1])
        axs2[i].set_title(plot_titles[i + 6], fontsize=title_size)
        axs2[i].tick_params(axis='both', labelsize=label_size)
        fig2.colorbar(im, ax=axs2[i]).ax.tick_params(labelsize=cbar_size)

    data8, cmap8 = plot_data[8]
    im8 = axs2[2].imshow(data8, cmap=cmap8, extent = [0, 1, 0, 1])
    axs2[2].set_title(plot_titles[8], fontsize=title_size)
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
    cmap1 = 'PuBuGn'
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
        im = axs1[0, i].imshow(data, cmap=cmap, vmin=row1_min, vmax=row1_max, extent=[-1,1,1,0])
        axs1[0, i].set_title(plot_titles[i], fontsize=title_size)
        axs1[0, i].tick_params(axis='both', labelsize=label_size)
        fig1.colorbar(im, ax=axs1[0, i]).ax.tick_params(labelsize=cbar_size)

    data2, cmap2 = plot_data[2]
    im2 = axs1[0, 2].imshow(data2, cmap=cmap2, extent=[-1,1,1,0])
    axs1[0, 2].set_title(plot_titles[2], fontsize=title_size)
    axs1[0, 2].tick_params(axis='both', labelsize=label_size)
    fig1.colorbar(im2, ax=axs1[0, 2]).ax.tick_params(labelsize=cbar_size)

    for i in range(2):
        data, cmap = plot_data[i + 3]
        im = axs1[1, i].imshow(data, cmap=cmap, vmin=row2_min, vmax=row2_max, extent=[-1,1,1,0])
        axs1[1, i].set_title(plot_titles[i + 3], fontsize=title_size)
        axs1[1, i].tick_params(axis='both', labelsize=label_size)
        fig1.colorbar(im, ax=axs1[1, i]).ax.tick_params(labelsize=cbar_size)

    data5, cmap5 = plot_data[5]
    im5 = axs1[1, 2].imshow(data5, cmap=cmap5, extent=[-1,1,1,0])
    axs1[1, 2].set_title(plot_titles[5], fontsize=title_size)
    axs1[1, 2].tick_params(axis='both', labelsize=label_size)
    fig1.colorbar(im5, ax=axs1[1, 2]).ax.tick_params(labelsize=cbar_size)

    plt.tight_layout()
    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f"part1_{added_text}_{current_time}.png")

    fig2, axs2 = plt.subplots(1, 3, figsize=(20, 3))

    for i in range(2):
        data, cmap = plot_data[i + 6]
        im = axs2[i].imshow(data, cmap=cmap, vmin=row3_min, vmax=row3_max, extent=[-1,1,1,0])
        axs2[i].set_title(plot_titles[i + 6], fontsize=title_size)
        axs2[i].tick_params(axis='both', labelsize=label_size)
        fig2.colorbar(im, ax=axs2[i]).ax.tick_params(labelsize=cbar_size)

    data8, cmap8 = plot_data[8]
    im8 = axs2[2].imshow(data8, cmap=cmap8, extent=[-1,1,1,0])
    axs2[2].set_title(plot_titles[8], fontsize=title_size)
    axs2[2].tick_params(axis='both', labelsize=label_size)
    fig2.colorbar(im8, ax=axs2[2]).ax.tick_params(labelsize=cbar_size)

    plt.tight_layout()
    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f"part2_{added_text}_{current_time}.png")