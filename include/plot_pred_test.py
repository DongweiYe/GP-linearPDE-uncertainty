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
    yvar_mean = jnp.mean(yvar_list, axis=0)
    return ymean_var + yvar_mean


def plot_prediction_results_test(X_plot_prediction, u_values_gt, y_final_mean_list_prior, y_final_var_list_prior,
                            y_final_mean_list_posterior, y_final_var_list_posterior):
    # 计算预测结果
    gp_mean_posterior = prediction_mean(y_final_mean_list_posterior).reshape(100, 100)
    abs_diff_gt_gp = jnp.abs(u_values_gt - gp_mean_posterior)
    var_prior = prediction_variance(y_final_mean_list_prior, y_final_var_list_prior).reshape(100, 100)
    var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior).reshape(100, 100)
    abs_var_diff = jnp.abs(var_prior - var_posterior)

    # 绘制 6 个子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 设置颜色映射
    cmap1 = 'GnBu'
    cmap2 = 'PuRd'

    # 设置字体大小
    font_kwargs = {'fontsize': 20}
    tick_fontsize = 18
    colorbar_fontsize = 18

    # tick 的间隔
    tick_spacing = 20

    # 绘制heatmap和colorbar
    def plot_heatmap(ax, data, title, cmap):
        im = ax.imshow(data, cmap=cmap, origin='lower')
        ax.set_title(title, **font_kwargs)
        ax.set_xlabel('x', **font_kwargs)
        ax.set_ylabel('t', **font_kwargs)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.xaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(plt.MultipleLocator(tick_spacing))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

    plot_heatmap(axes[0, 0], u_values_gt, 'Ground Truth', cmap1)
    plot_heatmap(axes[0, 1], gp_mean_posterior, 'GP Mean Prediction (Posterior)', cmap1)
    plot_heatmap(axes[0, 2], abs_diff_gt_gp, 'Absolute Difference (GT vs GP Mean)', cmap1)
    plot_heatmap(axes[1, 0], var_prior, 'Variance with Prior', cmap2)
    plot_heatmap(axes[1, 1], var_posterior, 'Variance with Posterior', cmap2)
    plot_heatmap(axes[1, 2], abs_var_diff, 'Absolute Variance Difference', cmap2)
    current_time = datetime.datetime.now().strftime("%H%M%S")
    plt.savefig(f"mean_var_{current_time}", bbox_inches='tight')
    plt.tight_layout()
    plt.show()


