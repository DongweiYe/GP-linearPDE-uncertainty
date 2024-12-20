# %%
import datetime
import gc
import os
import pickle

import jax
import jax.numpy as jnp
import jax.scipy.linalg as la
from jax.scipy import linalg

from include.heat2d import u_xt, compute_kuu, compute_kfu, compute_kuf, compute_kff
from include.main_loadForPred_rd import main_loadForPred_rd
from include.mcmc_posterior import compute_K
from include.plot_pred import plot_and_save_prediction_results, prediction_mean, prediction_variance, \
    plot_and_save_prediction_results_combine

os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_enable_x64", True)

current_time = datetime.datetime.now().strftime("%m%d")


def main_loadForPred():
    pred_mesh = 150
    text = "h.pkl"
    load_path = f"./results/load_data"
    # text = "chains1_f256_k0.8_assumption0.001_prior0.04_noise0.04_maxsamples2600_numpriorsamples_200_learnlr0.001&1000_0815.pkl"
    # load_path = f"./results/datas/trained_params/1219"

    print('start inference')
    print("pred_mesh:", pred_mesh, "\n")
    print("text:", text, "\n")

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
    Xf = variables['Xf']
    noise_std = variables['noise_std']
    prior_var = variables['prior_var']
    assumption_sigma = variables['assumption_sigma']
    k = variables['k']
    max_samples = variables['max_samples']
    num_chains = variables['num_chains']
    number_f = variables['number_f']
    posterior_samples_list = variables['posterior_samples_list']
    prior_samples = variables['prior_samples']
    Y = variables['Y']
    param_iter = variables['param_iter']
    Xu_fixed = variables['Xu_fixed']
    added_text = f"heat_Predction_f{number_f}_chains{num_chains}_k{k}_assumption{assumption_sigma}_noisestd{noise_std}_{prior_var}_k{k}_{max_samples}_{current_time}"


# %%
# # %%
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

        zz_cc = compute_kuu(Xcz, Xcz, params_kuu)
        zg_cc = compute_kuf(Xcz, Xcg, params, lengthscale_x, lengthscale_t)

        gz_cc = compute_kfu(Xcg, Xcz, params, lengthscale_x, lengthscale_t)
        gg_cc = compute_kff(Xcg, Xcg, params, lengthscale_x, lengthscale_t)
        K = jnp.block([[zz_cc, zg_cc], [gz_cc, gg_cc]])
        return K

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

    def is_symmetric(matrix, tol=1e-8):
        return jnp.allclose(matrix, matrix.T, atol=tol)


    def gp_predict_diagonal_batch(init, z_prior, Xcz, Xcg, y, x_star, batch_size=2000):
        print("Starting gp_predict_diagonal_batch function")
        params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
        params = init
        print("z_prior shape: ", z_prior.shape)
        print("Xcz shape: ", Xcz.shape)
        Xuc= jnp.concatenate((z_prior, Xcz))

        K = compute_K_no(init, Xuc, Xcg)
        print("Computed K matrix")
        print("K", K)

        # K_inv_y = la.solve(K, y, assume_a='pos')

        jitter_values = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
        for jitter in jitter_values:
            K_jittered = add_jitter(K, jitter)
            pos_def = is_positive_definite(K_jittered)
            cond_number = compute_condition_number(K_jittered)
            print(f"Jitter: {jitter} | Positive Definite: {pos_def} | Condition Number: {cond_number}")
            if pos_def and cond_number < 1e10:
                print("1e3")
                print("Jittered matrix is positive definite, condition number is acceptable less then 1e15.")
                break
        mu_star = []
        sigma_star_diag = []
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

        for i in range(0, x_star.shape[0], batch_size):
            x_star_batch = x_star[i:i + batch_size]

            Xz = jnp.concatenate((z_prior, Xcz))

            k_zz_c_star = compute_kuu(Xz, x_star_batch, params_kuu)
            k_gz_c_star = compute_kfu(Xcg, x_star_batch, params, params[0][1][0].item(), params[0][1][1].item())

            k_x_star_batch = jnp.vstack((k_zz_c_star, k_gz_c_star))
            mu_star_batch = jnp.dot(k_x_star_batch.T, K_inv_y)

            K_inv_k_x_star_batch = la.solve(K_jittered, k_x_star_batch, assume_a='pos')
            sigma_star_batch = compute_kuu(x_star_batch, x_star_batch, params_kuu) - jnp.dot(k_x_star_batch.T,
                                                                                             K_inv_k_x_star_batch)
            sigma_star_batch_diag = sigma_star_batch.diagonal()
            print("sigma_star_batch_diag: ", sigma_star_batch_diag)
            mu_star.append(mu_star_batch)
            sigma_star_diag.append(sigma_star_batch_diag)

        mu_star = jnp.concatenate(mu_star, axis=0)
        sigma_star_diag = jnp.concatenate(sigma_star_diag, axis=0).flatten()
        print("sigma_star_diag: ", sigma_star_diag)

        del K_inv_y, K, k_zz_c_star, k_gz_c_star, k_x_star_batch, K_inv_k_x_star_batch
        gc.collect()
        return mu_star.flatten(), sigma_star_diag



    for i in range(posterior_samples_list.shape[0]):
        Xu_sample = posterior_samples_list[i, :, :]
        print("Xu_sample.shape", Xu_sample.shape)

        y_final_mean, y_final_var = gp_predict_diagonal_batch(param_iter, Xu_sample, Xu_fixed, Xf, Y, X_plot_prediction)
        print("Prediction mean shape: ", y_final_mean.shape)
        print("Prediction variance shape: ", y_final_var.shape)

        y_final_mean_list_posterior.append(y_final_mean.T)
        y_final_var_list_posterior.append(y_final_var.T)

        del Xu_sample, y_final_mean, y_final_var

        gc.collect()
        jax.clear_caches()
        print("posterior memory cleaned up after iteration", i)

    prior_samples_reshaped = prior_samples
    print("prior_samples_reshaped shape: ", prior_samples_reshaped.shape)
    print("prior_samples_reshaped: ", prior_samples_reshaped)

    for i in range(prior_samples_reshaped.shape[0]):
        Xu_sample_prior = prior_samples_reshaped[i, :, :]

        y_final_mean_prior, y_final_var_prior = gp_predict_diagonal_batch(param_iter, Xu_sample_prior, Xu_fixed, Xf, Y,
                                                           X_plot_prediction)
        print("prior Prediction mean shape: ", y_final_mean_prior.shape)
        print("prior Prediction variance shape: ", y_final_var_prior.shape)

        y_final_mean_list_prior.append(y_final_mean_prior.T)
        y_final_var_list_prior.append(y_final_var_prior.T)

        del Xu_sample_prior, y_final_mean_prior, y_final_var_prior

        gc.collect()
        jax.clear_caches()
        print("prior memory cleaned up after iteration", i)

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

    y_final_mean_list_prior = jnp.array(y_final_mean_list_prior)
    y_final_var_list_prior = jnp.array(y_final_var_list_prior)
    print("prior Prediction mean shape: ", y_final_mean_list_prior.shape)
    print("prior Prediction variance shape: ", y_final_var_list_prior.shape)

    y_final_mean_posterior = prediction_mean(y_final_mean_list_posterior)

    y_final_var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior)
    y_final_mean_prior = prediction_mean(y_final_mean_list_prior)

    y_final_var_prior = prediction_variance(y_final_mean_list_prior, y_final_var_list_prior)
    print("final posterior Prediction mean shape: ", y_final_mean_posterior.shape)
    print("final posterior Prediction variance shape: ", y_final_var_posterior.shape)
    print("final prior Prediction mean shape: ", y_final_mean_prior.shape)
    print("final prior Prediction variance shape: ", y_final_var_prior.shape)
    print("-------------------end prediction-------------------")

    u_values_gt = u_xt(X_plot_prediction)
    print("u_values_gt shape: ", u_values_gt.shape)

    gp_mean_posterior = prediction_mean(y_final_mean_list_posterior)
    print("gp_mean_posterior shape: ", gp_mean_posterior.shape)
    gp_mean_posterior = gp_mean_posterior.reshape(pred_mesh, pred_mesh)
    print("gp_mean_posterior: ", gp_mean_posterior)

    u_values_gt = u_values_gt.reshape(pred_mesh, pred_mesh)
    print("u_values_gt: ", u_values_gt)
    print("y_final_mean_list_prior:", y_final_mean_list_prior)
    gp_mean_prior = prediction_mean(y_final_mean_list_prior)
    print("gp_mean_prior shape: ", gp_mean_prior.shape)
    print("gp_mean_prior: ", gp_mean_prior)
    gp_mean_prior = gp_mean_prior.reshape(pred_mesh, pred_mesh)


    abs_diff_prior = jnp.abs(u_values_gt - gp_mean_prior)
    abs_diff_gt_gp = jnp.abs(u_values_gt - gp_mean_posterior)
    print("abs_diff_prior shape: ", abs_diff_prior.shape)
    print("abs_diff_prior: ", abs_diff_prior)
    print("abs_diff_gt_gp shape: ", abs_diff_gt_gp.shape)
    print("abs_diff_gt_gp: ", abs_diff_gt_gp)

    var_prior = prediction_variance(y_final_mean_list_prior, y_final_var_list_prior)
    print("var_prior shape: ", var_prior.shape)
    print("y_final_mean_list_prior: ", y_final_mean_list_prior)
    print("y_final_var_list_prior: ", y_final_var_list_prior)
    print("var_prior: ", var_prior)
    var_prior = var_prior.reshape(pred_mesh, pred_mesh)
    print("var_prior: ", var_prior)

    var_posterior = prediction_variance(y_final_mean_list_posterior, y_final_var_list_posterior)
    print("var_posterior shape: ", var_posterior.shape)
    var_posterior = var_posterior.reshape(pred_mesh, pred_mesh)
    print("pred_mesh: ", pred_mesh)
    print("var_posterior: ", var_posterior)

    abs_var_diff = jnp.abs(var_prior - var_posterior)
    print("abs_var_diff: ", abs_var_diff)
    print("abs_var_diff shape: ", abs_var_diff.shape)

    print("y_final_mean_list_prior has NaN: ", jnp.isnan(y_final_mean_list_prior).any())
    print("y_final_var_list_prior has NaN: ", jnp.isnan(y_final_var_list_prior).any())
    print("y_final_mean_list_posterior has NaN: ", jnp.isnan(y_final_mean_list_posterior).any())
    print("y_final_var_list_posterior has NaN: ", jnp.isnan(y_final_var_list_posterior).any())
    save_variables(added_text, u_values_gt=u_values_gt,
                   gp_mean_prior=gp_mean_prior,
                   abs_diff_prior=abs_diff_prior,
                   gp_mean_posterior=gp_mean_posterior,
                   abs_diff_gt_gp=abs_diff_gt_gp,
                   var_prior=var_prior,
                   var_posterior=var_posterior,
                   abs_var_diff=abs_var_diff,
                   add_text=added_text)

    plot_and_save_prediction_results(u_values_gt,
                                     gp_mean_prior,
                                     abs_diff_prior,
                                    gp_mean_posterior,
                                    abs_diff_gt_gp,
                                    var_prior,
                                    var_posterior,
                                    abs_var_diff, added_text)
    plot_and_save_prediction_results_combine(u_values_gt,
                                             gp_mean_prior,
                                             abs_diff_prior,
                                             gp_mean_posterior,
                                             abs_diff_gt_gp,
                                             var_prior,
                                             var_posterior,
                                             abs_var_diff, added_text)

if __name__ == '__main__':
    main_loadForPred()