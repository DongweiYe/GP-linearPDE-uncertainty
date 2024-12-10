import os
import jax
import datetime

from include.heat2d import u_xt
from include.mcmc_posterior import *

import matplotlib.pyplot as plt

import jax.numpy as jnp

import pickle
import os
import jax.scipy.linalg as la
import gc

from jax.scipy import linalg


def plot_f_inference(pred_mesh, param_iter, Xu_fixed, Yu_fixed, Xf, yf, added_text):
    print("start prediction")
    x_prediction = jnp.linspace(0, 1, pred_mesh)
    t_prediction = jnp.linspace(0, 1, pred_mesh)

    X_prediction, T_prediction = jnp.meshgrid(x_prediction, t_prediction)

    X_plot_prediction = jnp.vstack([X_prediction.ravel(), T_prediction.ravel()]).T

    y_final_mean_list_posterior = []
    y_final_var_list_posterior = []

    y_final_mean_list_prior = []
    y_final_var_list_prior = []


    def compute_K_no(init, Xcz, Xcg):
        params = init
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        lengthscale_x = params[0][1][0].item()
        lengthscale_t = params[0][1][1].item()
        # zz_uu = compute_kuu(Xuz, Xuz, params_kuu)
        # zz_uc = compute_kuu(Xuz, Xcz, params_kuu)
        # zg_uc = compute_kuf(Xuz, Xcg, params, lengthscale_x, lengthscale_t)
        # zz_cu = compute_kuu(Xcz, Xuz, params_kuu)
        zz_cc = compute_kuu(Xcz, Xcz, params_kuu)

        zg_cc = compute_kuf(Xcz, Xcg, params, lengthscale_x, lengthscale_t)
        # gz_cu = compute_kfu(Xcg, Xuz, params, lengthscale_x, lengthscale_t)
        gz_cc = compute_kfu(Xcg, Xcz, params, lengthscale_x, lengthscale_t)
        gg_cc = compute_kff(Xcg, Xcg, params, lengthscale_x, lengthscale_t)
        K = jnp.block([[zz_cc, zg_cc], [gz_cc, gg_cc]])
        return K


    def is_symmetric(matrix, tol=1e-8):
        return jnp.allclose(matrix, matrix.T, atol=tol)


    def compute_condition_number(matrix):
        singular_values = jnp.linalg.svd(matrix, compute_uv=False)
        cond_number = singular_values.max() / singular_values.min()
        return cond_number


    def is_positive_definite(matrix):
        try:
            jnp.linalg.cholesky(matrix)
            return True
        except jnp.linalg.LinAlgError:
            return False


    def add_jitter(matrix, jitter=1e-6):
        jitter_matrix = matrix + jitter * jnp.eye(matrix.shape[0])
        return jitter_matrix





    def gp_predict_diagonal_batch_no(init, Xcz, Xcg, y, x_star, batch_size=2000):
        print("Starting gp_predict_diagonal_batch function")
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        params = init
        K = compute_K_no(init, Xcz, Xcg)
        print("Computed K matrix")
        print("K", K)

        jitter_values = [1e-8, 1e-6, 1e-4, 1e-2]
        for jitter in jitter_values:
            K_jittered = add_jitter(K, jitter)
            pos_def = is_positive_definite(K_jittered)
            cond_number = compute_condition_number(K_jittered)
            print(f"Jitter: {jitter} | Positive Definite: {pos_def} | Condition Number: {cond_number}")
            if pos_def and cond_number < 1e6:
                break

        mu_star = []
        sigma_star_diag = []
        K_jittered = add_jitter(K, jitter)
        try:
            K_inv_y = linalg.solve(K_jittered, y, assume_a='pos')
            print("Solved K_inv_y successfully.")
        except Exception as e:
            print(f"Error in solving linear system: {e}")

        if jnp.isnan(K_inv_y).any() or jnp.isinf(K_inv_y).any():
            print("Result contains NaN or Inf values.")
        else:
            print("Result is valid.")
        symmetric = is_symmetric(K)
        print(f"Is K symmetric? {symmetric}")
        cond_number = compute_condition_number(K)
        print(f"Condition number of K: {cond_number}")
        # K_inv_y = la.solve(K, y, assume_a='pos')
        # print("K_inv_y ", K_inv_y )
        pos_def = is_positive_definite(K)
        print(f"Is K positive definite? {pos_def}")

        # K_inv_y = linalg.solve(K, y, assume_a='pos')

        for i in range(0, x_star.shape[0], batch_size):
            x_star_batch = x_star[i:i + batch_size]

            # k_zz_u_star = compute_kuu(z_prior, x_star_batch, params_kuu)
            k_zz_c_star = compute_kuu(Xcz, x_star_batch, params_kuu)
            k_gz_c_star = compute_kfu(Xcg, x_star_batch, params, params[0][1][0].item(), params[0][1][1].item())

            k_x_star_batch = jnp.vstack((k_zz_c_star, k_gz_c_star))
            mu_star_batch = jnp.dot(k_x_star_batch.T, K_inv_y)

            K_inv_k_x_star_batch = la.solve(K, k_x_star_batch, assume_a='pos')
            sigma_star_batch = compute_kuu(x_star_batch, x_star_batch, params_kuu) - jnp.dot(k_x_star_batch.T,
                                                                                             K_inv_k_x_star_batch)
            sigma_star_batch_diag = sigma_star_batch.diagonal()

            mu_star.append(mu_star_batch)
            sigma_star_diag.append(sigma_star_batch_diag)

        mu_star = jnp.concatenate(mu_star, axis=0)
        sigma_star_diag = jnp.concatenate(sigma_star_diag, axis=0).flatten()

        del K_inv_y, K, k_zz_c_star, k_gz_c_star, k_x_star_batch, K_inv_k_x_star_batch
        gc.collect()
        return mu_star.flatten(), sigma_star_diag


    Y_no = jnp.concatenate((Yu_fixed, yf))
    y_final_mean, y_final_var = gp_predict_diagonal_batch_no(param_iter, Xu_fixed, Xf, Y_no, X_plot_prediction)
    print("Prediction mean shape: ", y_final_mean.shape)
    print("Prediction variance shape: ", y_final_var.shape)

    y_final_mean_list_posterior.append(y_final_mean.T)
    y_final_var_list_posterior.append(y_final_var.T)

    gc.collect()
    jax.clear_caches()


    def save_variables(added_text, **variables):
        root_folder = "."
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        filename = f"Pred_{added_text}.pkl"
        file_path = os.path.join(root_folder, filename)

        with open(file_path, 'wb') as f:
            pickle.dump(variables, f)
        print(f"Variables saved to {file_path}")


    y_final_mean_list_posterior = jnp.array(y_final_mean_list_posterior)
    y_final_var_list_posterior = jnp.array(y_final_var_list_posterior)

    print("posterior Prediction mean shape: ", y_final_mean_list_posterior.shape)
    print("posterior Prediction variance shape: ", y_final_var_list_posterior.shape)

    # y_final_mean_posterior = prediction_mean(y_final_mean_list_posterior)

    # y_final_var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior)

    print("-------------------end prediction-------------------")

    u_values_gt = u_xt(X_plot_prediction)

    gp_mean_posterior = y_final_mean_list_posterior.reshape(pred_mesh, pred_mesh)
    u_values_gt = u_values_gt.reshape(pred_mesh, pred_mesh)


    # abs_diff_gt_gp = jnp.abs(u_values_gt - gp_mean_posterior)
    #
    # var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior).reshape(pred_mesh,
    #                                                                                                      pred_mesh)
    #

    def plot_combine(Xf, u_values_gt,
                     gp_mean_posterior,
                     added_text):

        plot_titles = [
            'Ground Truth',
            'GP prediction',
            'Absolute Error',
        ]
        cmap1 = 'GnBu'
        cmap2 = 'viridis'
        plot_data = [
            (u_values_gt, cmap1),
            (gp_mean_posterior, cmap1),
            (jnp.abs(u_values_gt - gp_mean_posterior), cmap1),  # 绝对误差

        ]

        row1_min = min(jnp.min(plot_data[0][0]), jnp.min(plot_data[1][0]))
        row1_max = max(jnp.max(plot_data[0][0]), jnp.max(plot_data[1][0]))

        # row2_min = min(jnp.min(plot_data[3][0]), jnp.min(plot_data[4][0]), jnp.min(plot_data[5][0]))
        # row2_max = max(jnp.max(plot_data[3][0]), jnp.max(plot_data[4][0]), jnp.max(plot_data[5][0]))

        fig, axs = plt.subplots(1, 3, figsize=(24, 6))

        for i in range(2):
            data, cmap = plot_data[i]
            im = axs[i].imshow(data, vmin=row1_min, vmax=row1_max, extent=[0, 1, 0, 1])
            axs[i].set_title(plot_titles[i])

            fig.colorbar(im, ax=axs[i])
        axs[0].scatter(Xf[:, 0], Xf[:, 1], color="red")

        data2, cmap2 = plot_data[2]
        im2 = axs[2].imshow(data2, extent=[0, 1, 0, 1])
        axs[2].set_title(plot_titles[2])
        fig.colorbar(im2, ax=axs[2])

        # for i in range(3):
        #     data, cmap = plot_data[i + 3]
        #     im = axs[1, i].imshow(data, cmap=cmap, vmin=row2_min, vmax=row2_max)
        #     axs[1, i].set_title(plot_titles[i + 3])
        #     fig.colorbar(im, ax=axs[1, i])

        plt.tight_layout()
        current_time = datetime.datetime.now().strftime("%M%S")
        plt.savefig(f"combined_plot_{added_text}_{current_time}.png")
        plt.show()


    plot_combine(Xf, u_values_gt,
                 gp_mean_posterior,
                 added_text)

    print(gp_mean_posterior.shape)
    print(u_values_gt.shape)


def plot_f_inference_rd_init(param_iter, Xu_fixed, Yu_fixed, Xf, yf, added_text,  X_plot_prediction, data, learning,init):
    print("start prediction")
    # x_prediction = jnp.linspace(0, 1, pred_mesh)
    # t_prediction = jnp.linspace(0, 1, pred_mesh)

    # X_prediction, T_prediction = jnp.meshgrid(x_prediction, t_prediction)

    # X_plot_prediction = jnp.vstack([X_prediction.ravel(), T_prediction.ravel()]).T

    y_final_mean_list_posterior = []
    y_final_var_list_posterior = []

    y_final_mean_list_prior = []
    y_final_var_list_prior = []

    # jitter = 1e-4
    # print("jitter: ", jitter)
    def compute_K_no(init, Xcz, Xcg):
        params = init
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        lengthscale_x = params[0][1][0].item()
        lengthscale_t = params[0][1][1].item()
        # zz_uu = compute_kuu(Xuz, Xuz, params_kuu)
        # zz_uc = compute_kuu(Xuz, Xcz, params_kuu)
        # zg_uc = compute_kuf(Xuz, Xcg, params, lengthscale_x, lengthscale_t)
        # zz_cu = compute_kuu(Xcz, Xuz, params_kuu)
        zz_cc = compute_kuu_rd(Xcz, Xcz, params_kuu)
        # zz_cc = add_jitter(zz_cc, jitter)

        zg_cc = compute_kuf_rd(Xcz, Xcg, params, lengthscale_x, lengthscale_t)
        # gz_cu = compute_kfu(Xcg, Xuz, params, lengthscale_x, lengthscale_t)
        gz_cc = compute_kfu_rd(Xcg, Xcz, params, lengthscale_x, lengthscale_t)
        gg_cc = compute_kff_rd(Xcg, Xcg, params, lengthscale_x, lengthscale_t)
        K = jnp.block([[zz_cc, zg_cc], [gz_cc, gg_cc]])
        return K


    def is_symmetric(matrix, tol=1e-8):
        return jnp.allclose(matrix, matrix.T, atol=tol)


    def compute_condition_number(matrix):
        singular_values = jnp.linalg.svd(matrix, compute_uv=False)
        cond_number = singular_values.max() / singular_values.min()
        return cond_number


    def is_positive_definite(matrix):
        try:
            jnp.linalg.cholesky(matrix)
            return True
        except jnp.linalg.LinAlgError:
            return False


    def add_jitter(matrix, jitter=1e-6):
        jitter_matrix = matrix + jitter * jnp.eye(matrix.shape[0])
        return jitter_matrix





    def gp_predict_diagonal_batch_no(init, Xcz, Xcg, y, x_star, batch_size=2000):
        print("Starting gp_predict_diagonal_batch function")
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        params = init
        K = compute_K_no(init, Xcz, Xcg)
        print("Computed K matrix")
        print("K", K)

        jitter_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        for jitter in jitter_values:
            K_jittered = add_jitter(K, jitter)
            pos_def = is_positive_definite(K_jittered)
            cond_number = compute_condition_number(K_jittered)
            print(f"Jitter: {jitter} | Positive Definite: {pos_def} | Condition Number: {cond_number}")
            if pos_def and cond_number < 1e7:
                break

        mu_star = []
        sigma_star_diag = []
        K_jittered = add_jitter(K, jitter)
        try:
            K_inv_y = linalg.solve(K_jittered, y, assume_a='pos')
            print("Solved K_inv_y successfully.")
        except Exception as e:
            print(f"Error in solving linear system: {e}")

        if jnp.isnan(K_inv_y).any() or jnp.isinf(K_inv_y).any():
            print("Result contains NaN or Inf values.")
        else:
            print("Result is valid.")
        symmetric = is_symmetric(K)
        print(f"Is K symmetric? {symmetric}")
        cond_number = compute_condition_number(K)
        print(f"Condition number of K: {cond_number}")

        pos_def = is_positive_definite(K)
        print(f"Is K positive definite? {pos_def}")

        # K_inv_y = la.solve(K, y, assume_a='pos')
        # print("K_inv_y ", K_inv_y )
        # print("no k inv y")
        # print("no k inv y")
        # print("no k inv y")
        # print("no k inv y")
        # print("no k inv y")
        # 327330 print("no k inv y original train")
        # 32729 print("k inv y original train")

        # 32728 print("k inv y train")

        # 327338 print("no k inv y train")
        # 32739 print("k inv y  train")
        # 3273 print("no k inv y original train")




        for i in range(0, x_star.shape[0], batch_size):
            x_star_batch = x_star[i:i + batch_size]

            # k_zz_u_star = compute_kuu(z_prior, x_star_batch, params_kuu)
            k_zz_c_star = compute_kuu_rd(Xcz, x_star_batch, params_kuu)
            k_gz_c_star = compute_kfu_rd(Xcg, x_star_batch, params, params[0][1][0].item(), params[0][1][1].item())

            k_x_star_batch = jnp.vstack((k_zz_c_star, k_gz_c_star))
            mu_star_batch = jnp.dot(k_x_star_batch.T, K_inv_y)

            K_inv_k_x_star_batch = la.solve(K, k_x_star_batch, assume_a='pos')
            sigma_star_batch = compute_kuu_rd(x_star_batch, x_star_batch, params_kuu) - jnp.dot(k_x_star_batch.T,
                                                                                             K_inv_k_x_star_batch)
            sigma_star_batch_diag = sigma_star_batch.diagonal()

            mu_star.append(mu_star_batch)
            sigma_star_diag.append(sigma_star_batch_diag)

        mu_star = jnp.concatenate(mu_star, axis=0)
        sigma_star_diag = jnp.concatenate(sigma_star_diag, axis=0).flatten()

        del K_inv_y, K, k_zz_c_star, k_gz_c_star, k_x_star_batch, K_inv_k_x_star_batch
        gc.collect()
        return mu_star.flatten(), sigma_star_diag


    Y_no = jnp.concatenate((Yu_fixed, yf))
    y_final_mean, y_final_var = gp_predict_diagonal_batch_no(param_iter, Xu_fixed, Xf, Y_no, X_plot_prediction)
    print("Prediction mean shape: ", y_final_mean.shape)
    print("Prediction variance shape: ", y_final_var.shape)

    y_final_mean_list_posterior.append(y_final_mean.T)
    y_final_var_list_posterior.append(y_final_var.T)

    gc.collect()
    jax.clear_caches()


    def save_variables(added_text, **variables):
        root_folder = "."
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        filename = f"Pred_{added_text}.pkl"
        file_path = os.path.join(root_folder, filename)

        with open(file_path, 'wb') as f:
            pickle.dump(variables, f)
        print(f"Variables saved to {file_path}")


    y_final_mean_list_posterior = jnp.array(y_final_mean_list_posterior)
    y_final_var_list_posterior = jnp.array(y_final_var_list_posterior)

    print("posterior Prediction mean shape: ", y_final_mean_list_posterior.shape)
    print("posterior Prediction variance shape: ", y_final_var_list_posterior.shape)

    # y_final_mean_posterior = prediction_mean(y_final_mean_list_posterior)

    # y_final_var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior)

    print("-------------------end prediction-------------------")

    # u_values_gt = u_xt(X_plot_prediction)

    gp_mean_posterior = y_final_mean_list_posterior.reshape(data.shape)
    u_values_gt = data
    print("gp_mean_posterior shape: ", gp_mean_posterior.shape)
    print("u_values_gt shape: ", u_values_gt.shape)


    # abs_diff_gt_gp = jnp.abs(u_values_gt - gp_mean_posterior)
    #
    # var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior).reshape(pred_mesh,
    #                                                                                                      pred_mesh)
    #

    def plot_combine(Xf, u_values_gt,
                     gp_mean_posterior,
                     added_text, learning):

        plot_titles = [
            'Ground Truth',
            'GP Prediction',
            'Absolute Error',
        ]
        cmap2 = 'viridis'
        cmap1 = 'GnBu'
        plot_data = [
            (u_values_gt, cmap1),
            (gp_mean_posterior, cmap1),
            (jnp.abs(u_values_gt - gp_mean_posterior), cmap1),

        ]

        row1_min = jnp.min(plot_data[0][0])
        row1_max = jnp.max(plot_data[0][0])

        # row1_min = min(jnp.min(plot_data[0][0]), jnp.min(plot_data[1][0]))
        # row1_max = max(jnp.max(plot_data[0][0]), jnp.max(plot_data[1][0]))

        # row2_min = min(jnp.min(plot_data[3][0]), jnp.min(plot_data[4][0]), jnp.min(plot_data[5][0]))
        # row2_max = max(jnp.max(plot_data[3][0]), jnp.max(plot_data[4][0]), jnp.max(plot_data[5][0]))

        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        shrink = 0.4

        for i in range(2):
            data, cmap = plot_data[i]
            im = axs[i].imshow(data, vmin=row1_min, vmax=row1_max, extent=[-1,1,1,0])
            axs[i].set_title(plot_titles[i])
            #axs[i].scatter(Xf[:, 0], Xf[:, 1], color="red")
            fig.colorbar(im, ax=axs[i], shrink=shrink)

        # for i in range(3):
        #     data, cmap = plot_data[i + 3]
        #     im = axs[1, i].imshow(data, cmap=cmap, vmin=row2_min, vmax=row2_max)
        #     axs[1, i].set_title(plot_titles[i + 3])
        #     fig.colorbar(im, ax=axs[1, i])

        axs[0].scatter(Xf[:, 0], Xf[:, 1], color="red")

        data2, cmap2 = plot_data[2]
        im2 = axs[2].imshow(data2, extent=[-1,1,1,0])
        axs[2].set_title(plot_titles[2])
        fig.colorbar(im2, ax=axs[2], shrink=shrink)

        fig.text(0.01, 0.93, f"init={init}", ha='left', fontsize=16)
        fig.text(0.01, 0.91, f"param_iter={param_iter} ", ha='left', fontsize=16)

        plt.tight_layout()
        current_time = datetime.datetime.now().strftime("%M%S")
        plt.savefig(f"combined_plot_{added_text}_{current_time}_{learning}.png")

        plt.show()


    plot_combine(Xf, u_values_gt,
                 gp_mean_posterior,
                 added_text,learning)

    print("gp_mean_posterior:", gp_mean_posterior)
    print("u_values_gt:", u_values_gt)


def plot_f_inference_rd(param_iter, Xu_fixed, Yu_fixed, Xf, yf, added_text,  X_plot_prediction, data, learning):
    print("start prediction")
    # x_prediction = jnp.linspace(0, 1, pred_mesh)
    # t_prediction = jnp.linspace(0, 1, pred_mesh)

    # X_prediction, T_prediction = jnp.meshgrid(x_prediction, t_prediction)

    # X_plot_prediction = jnp.vstack([X_prediction.ravel(), T_prediction.ravel()]).T

    y_final_mean_list_posterior = []
    y_final_var_list_posterior = []

    y_final_mean_list_prior = []
    y_final_var_list_prior = []
    def add_jitter(matrix, jitter=1e-6):
        print("add_jitter")
        print("jitter: ", jitter)
        jitter_matrix = matrix + jitter * jnp.eye(matrix.shape[0])
        return jitter_matrix

    def compute_K_no(init, Xcz, Xcg):
        params = init
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        lengthscale_x = params[0][1][0].item()
        lengthscale_t = params[0][1][1].item()
        # zz_uu = compute_kuu(Xuz, Xuz, params_kuu)
        # zz_uc = compute_kuu(Xuz, Xcz, params_kuu)
        # zg_uc = compute_kuf(Xuz, Xcg, params, lengthscale_x, lengthscale_t)
        # zz_cu = compute_kuu(Xcz, Xuz, params_kuu)
        zz_cc = compute_kuu_rd(Xcz, Xcz, params_kuu)
        zz_cc = add_jitter(zz_cc)

        zg_cc = compute_kuf_rd(Xcz, Xcg, params, lengthscale_x, lengthscale_t)
        # gz_cu = compute_kfu(Xcg, Xuz, params, lengthscale_x, lengthscale_t)
        gz_cc = compute_kfu_rd(Xcg, Xcz, params, lengthscale_x, lengthscale_t)
        gg_cc = compute_kff_rd(Xcg, Xcg, params, lengthscale_x, lengthscale_t)
        K = jnp.block([[zz_cc, zg_cc], [gz_cc, gg_cc]])
        return K


    def is_symmetric(matrix, tol=1e-8):
        return jnp.allclose(matrix, matrix.T, atol=tol)


    def compute_condition_number(matrix):
        singular_values = jnp.linalg.svd(matrix, compute_uv=False)
        cond_number = singular_values.max() / singular_values.min()
        return cond_number


    def is_positive_definite(matrix):
        try:
            jnp.linalg.cholesky(matrix)
            return True
        except jnp.linalg.LinAlgError:
            return False








    def gp_predict_diagonal_batch_no(init, Xcz, Xcg, y, x_star, batch_size=2000):
        print("Starting gp_predict_diagonal_batch function")
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        params = init
        K = compute_K_no(init, Xcz, Xcg)
        print("Computed K matrix")
        print("K", K)

        # jitter_values = [1e-8, 1e-6, 1e-4, 1e-2]
        # for jitter in jitter_values:
        #     K_jittered = add_jitter(K, jitter)
        #     pos_def = is_positive_definite(K_jittered)
        #     cond_number = compute_condition_number(K_jittered)
        #     print(f"Jitter: {jitter} | Positive Definite: {pos_def} | Condition Number: {cond_number}")
        #     if pos_def and cond_number < 1e6:
        #         break
        #
        mu_star = []
        sigma_star_diag = []
        # K_jittered = add_jitter(K, jitter)
        # try:
        #     K_inv_y = linalg.solve(K_jittered, y, assume_a='pos')
        #     print("Solved K_inv_y successfully.")
        # except Exception as e:
        #     print(f"Error in solving linear system: {e}")
        #
        # if jnp.isnan(K_inv_y).any() or jnp.isinf(K_inv_y).any():
        #     print("Result contains NaN or Inf values.")
        # else:
        #     print("Result is valid.")
        # symmetric = is_symmetric(K)
        # print(f"Is K symmetric? {symmetric}")
        # cond_number = compute_condition_number(K)
        # print(f"Condition number of K: {cond_number}")

        K_inv_y = la.solve(K, y, assume_a='pos')
        print("K_inv_y ", K_inv_y )
        pos_def = is_positive_definite(K)
        print(f"Is K positive definite? {pos_def}")

        for i in range(0, x_star.shape[0], batch_size):
            x_star_batch = x_star[i:i + batch_size]

            # k_zz_u_star = compute_kuu(z_prior, x_star_batch, params_kuu)
            k_zz_c_star = compute_kuu_rd(Xcz, x_star_batch, params_kuu)
            k_gz_c_star = compute_kfu_rd(Xcg, x_star_batch, params, params[0][1][0].item(), params[0][1][1].item())

            k_x_star_batch = jnp.vstack((k_zz_c_star, k_gz_c_star))
            mu_star_batch = jnp.dot(k_x_star_batch.T, K_inv_y)

            K_inv_k_x_star_batch = la.solve(K, k_x_star_batch, assume_a='pos')
            sigma_star_batch = compute_kuu_rd(x_star_batch, x_star_batch, params_kuu) - jnp.dot(k_x_star_batch.T,
                                                                                             K_inv_k_x_star_batch)
            sigma_star_batch_diag = sigma_star_batch.diagonal()

            mu_star.append(mu_star_batch)
            sigma_star_diag.append(sigma_star_batch_diag)

        mu_star = jnp.concatenate(mu_star, axis=0)
        sigma_star_diag = jnp.concatenate(sigma_star_diag, axis=0).flatten()

        del K_inv_y, K, k_zz_c_star, k_gz_c_star, k_x_star_batch, K_inv_k_x_star_batch
        gc.collect()
        return mu_star.flatten(), sigma_star_diag


    Y_no = jnp.concatenate((Yu_fixed, yf))
    y_final_mean, y_final_var = gp_predict_diagonal_batch_no(param_iter, Xu_fixed, Xf, Y_no, X_plot_prediction)
    print("Prediction mean shape: ", y_final_mean.shape)
    print("Prediction variance shape: ", y_final_var.shape)

    y_final_mean_list_posterior.append(y_final_mean.T)
    y_final_var_list_posterior.append(y_final_var.T)

    gc.collect()
    jax.clear_caches()


    def save_variables(added_text, **variables):
        root_folder = "."
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        filename = f"Pred_{added_text}.pkl"
        file_path = os.path.join(root_folder, filename)

        with open(file_path, 'wb') as f:
            pickle.dump(variables, f)
        print(f"Variables saved to {file_path}")


    y_final_mean_list_posterior = jnp.array(y_final_mean_list_posterior)
    y_final_var_list_posterior = jnp.array(y_final_var_list_posterior)

    print("posterior Prediction mean shape: ", y_final_mean_list_posterior.shape)
    print("posterior Prediction variance shape: ", y_final_var_list_posterior.shape)

    # y_final_mean_posterior = prediction_mean(y_final_mean_list_posterior)

    # y_final_var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior)

    print("-------------------end prediction-------------------")

    # u_values_gt = u_xt(X_plot_prediction)

    gp_mean_posterior = y_final_mean_list_posterior.reshape(data.shape)
    u_values_gt = data
    print("gp_mean_posterior shape: ", gp_mean_posterior.shape)
    print("u_values_gt shape: ", u_values_gt.shape)


    # abs_diff_gt_gp = jnp.abs(u_values_gt - gp_mean_posterior)
    #
    # var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior).reshape(pred_mesh,
    #                                                                                                      pred_mesh)
    #

    def plot_combine(Xf, u_values_gt,
                     gp_mean_posterior,
                     added_text, learning):

        plot_titles = [
            'Ground Truth',
            'GP Prediction',
            'Absolute Error',
        ]
        cmap2 = 'viridis'
        cmap1 = 'GnBu'
        plot_data = [
            (u_values_gt, cmap1),
            (gp_mean_posterior, cmap1),
            (jnp.abs(u_values_gt - gp_mean_posterior), cmap1),

        ]

        row1_min = min(jnp.min(plot_data[0][0]), jnp.min(plot_data[1][0]))
        row1_max = max(jnp.max(plot_data[0][0]), jnp.max(plot_data[1][0]))

        # row2_min = min(jnp.min(plot_data[3][0]), jnp.min(plot_data[4][0]), jnp.min(plot_data[5][0]))
        # row2_max = max(jnp.max(plot_data[3][0]), jnp.max(plot_data[4][0]), jnp.max(plot_data[5][0]))

        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        shrink = 0.4

        for i in range(2):
            data, cmap = plot_data[i]
            im = axs[i].imshow(data, vmin=row1_min, vmax=row1_max, extent=[-1,1,1,0])
            axs[i].set_title(plot_titles[i])
            #axs[i].scatter(Xf[:, 0], Xf[:, 1], color="red")
            fig.colorbar(im, ax=axs[i], shrink=shrink)

        # for i in range(3):
        #     data, cmap = plot_data[i + 3]
        #     im = axs[1, i].imshow(data, cmap=cmap, vmin=row2_min, vmax=row2_max)
        #     axs[1, i].set_title(plot_titles[i + 3])
        #     fig.colorbar(im, ax=axs[1, i])

        axs[0].scatter(Xf[:, 0], Xf[:, 1], color="red")

        data2, cmap2 = plot_data[2]
        im2 = axs[2].imshow(data2, extent=[-1,1,1,0])
        axs[2].set_title(plot_titles[2])
        fig.colorbar(im2, ax=axs[2], shrink=shrink)

        plt.tight_layout()
        current_time = datetime.datetime.now().strftime("%M%S")
        plt.savefig(f"combined_plot_{added_text}_{current_time}_{learning}.png")
        plt.show()


    plot_combine(Xf, u_values_gt,
                 gp_mean_posterior,
                 added_text,learning)

    print("gp_mean_posterior:", gp_mean_posterior)
    print("u_values_gt:", u_values_gt)