import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import BarkerMH
from numpyro.infer.mcmc import MCMC

from .heat2d import compute_kuu, compute_kuf, compute_kfu, \
    compute_kff, compute_kuu_rd, compute_kuf_rd, compute_kfu_rd, compute_kff_rd


def add_jitter(matrix, jitter=1e-2):
    jitter_matrix = matrix + jitter * jnp.eye(matrix.shape[0])
    print("jitter nmuber is:", jitter)
    return jitter_matrix


def compute_K(init, z_prior, Xcz, Xcg):
    Xuz = z_prior
    params = init
    jitter_u = 1e-8
    params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    lengthscale_x = params[0][1][0].item()
    lengthscale_t = params[0][1][1].item()
    zz_uu = compute_kuu(Xuz, Xuz, params_kuu)
    zz_uu = add_jitter(zz_uu, jitter_u)
    zz_uc = compute_kuu(Xuz, Xcz, params_kuu)
    zg_uc = compute_kuf(Xuz, Xcg, params, lengthscale_x, lengthscale_t)
    zz_cu = compute_kuu(Xcz, Xuz, params_kuu)
    zz_cc = compute_kuu(Xcz, Xcz, params_kuu)
    zg_cc = compute_kuf(Xcz, Xcg, params, lengthscale_x, lengthscale_t)
    gz_cu = compute_kfu(Xcg, Xuz, params, lengthscale_x, lengthscale_t)
    gz_cc = compute_kfu(Xcg, Xcz, params, lengthscale_x, lengthscale_t)
    gg_cc = compute_kff(Xcg, Xcg, params, lengthscale_x, lengthscale_t)
    K = jnp.block([[zz_uu, zz_uc, zg_uc], [zz_cu, zz_cc, zg_cc], [gz_cu, gz_cc, gg_cc]])
    return K


def compute_K_rd(init, z_prior, Xcz, Xcg):
    Xuz = z_prior
    params = init
    params_kuu = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    lengthscale_x = params[0][1][0].item()
    lengthscale_t = params[0][1][1].item()
    zz_uu = compute_kuu_rd(Xuz, Xuz, params_kuu)
    jitter_f = 1e-4 #1e-7
    zz_uu = add_jitter(zz_uu, jitter_f)
    zz_uc = compute_kuu_rd(Xuz, Xcz, params_kuu)
    zg_uc = compute_kuf_rd(Xuz, Xcg, params, lengthscale_x, lengthscale_t)
    zz_cu = compute_kuu_rd(Xcz, Xuz, params_kuu)
    zz_cc = compute_kuu_rd(Xcz, Xcz, params_kuu)
    zg_cc = compute_kuf_rd(Xcz, Xcg, params, lengthscale_x, lengthscale_t)
    gz_cu = compute_kfu_rd(Xcg, Xuz, params, lengthscale_x, lengthscale_t)
    gz_cc = compute_kfu_rd(Xcg, Xcz, params, lengthscale_x, lengthscale_t)
    gg_cc = compute_kff_rd(Xcg, Xcg, params, lengthscale_x, lengthscale_t)
    K = jnp.block([[zz_uu, zz_uc, zg_uc], [zz_cu, zz_cc, zg_cc], [gz_cu, gz_cc, gg_cc]])
    print("compute_K_rd shape:", K.shape)
    return K


def log_prior(x, mean, cov):
    dim = mean.shape[0]
    return -0.5 * ((x - mean).T @ jnp.linalg.inv(cov) @ (x - mean)) - 0.5 * dim * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(
        jnp.linalg.det(cov))


def log_likelihood(y, cov):
    dim = y.shape[0]
    return -0.5 * (y.T @ jnp.linalg.inv(cov) @ y) - 0.5 * dim * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(jnp.linalg.det(cov))


def single_component_metropolis_hasting(rng_key, max_samples, assumption_sigma, prior_mean, prior_cov, init, Xfz, Xfg,
                                        Y, k, num_x_only):
    prior_mean_flat = jnp.ravel(prior_mean)
    prior_cov_flat = jnp.kron(prior_cov, jnp.eye(prior_mean.shape[0]))  # shape (2*num_vague, 2*num_vague)

    num_vague = prior_mean.shape[0]
    num_x_t_noise = num_vague - num_x_only

    xvague_sample_current = prior_mean_flat
    xvague_sample_list = jnp.empty((0, prior_mean_flat.shape[0]), dtype=prior_mean_flat.dtype)

    current_log_prob = dist.MultivariateNormal(prior_mean_flat, covariance_matrix=prior_cov_flat).log_prob(
        xvague_sample_current)

    num_warmup = int(max_samples * k)
    print(f"Setting {k} of max_samples as warm-up")
    acceptance_count = 0
    print("num_warmup:", num_warmup)
    acceptance_flags = []

    target_acceptance_rate = 0.3
    tolerance = 0.05
    adjustment_interval = 100

    for i in range(max_samples - 1):
        for j in range(num_x_t_noise * 2 + num_x_only):
            rng_key, key_new = random.split(rng_key)
            step_size = assumption_sigma

            if j < num_x_t_noise * 2:
                x_new = xvague_sample_current.at[j].set(
                    random.normal(key_new, shape=()) * step_size + xvague_sample_current[j]
                )
            else:
                index_x = num_x_t_noise * 2 + (j - num_x_t_noise * 2) * 2
                x_new = xvague_sample_current.at[index_x].set(
                    random.normal(key_new, shape=()) * step_size + xvague_sample_current[index_x]
                )

            x_new = jnp.maximum(jnp.minimum(1, x_new), 0)  # Non-negative

            x_new_reshape = x_new.reshape(prior_mean.shape)
            xvague_sample_current_reshape = xvague_sample_current.reshape(prior_mean.shape)

            p_x_prior_new_log = dist.MultivariateNormal(prior_mean_flat, covariance_matrix=prior_cov_flat).log_prob(
                x_new)
            p_x_prior_current_log = dist.MultivariateNormal(prior_mean_flat, covariance_matrix=prior_cov_flat).log_prob(
                xvague_sample_current)
            cov_current = compute_K(init, xvague_sample_current_reshape, Xfz, Xfg)
            cov_new = compute_K(init, x_new_reshape, Xfz, Xfg)

            Y_ravel = Y.ravel()
            p_y_likelihood_new_log = dist.MultivariateNormal(jnp.zeros(Y_ravel.shape),
                                                             covariance_matrix=cov_new).log_prob(Y_ravel)
            p_y_likelihood_current_log = dist.MultivariateNormal(jnp.zeros(Y_ravel.shape),
                                                                 covariance_matrix=cov_current).log_prob(Y_ravel)

            print("p_x_prior_new_log:", p_x_prior_new_log)
            print("p_x_prior_current_log:", p_x_prior_current_log)
            print("p_y_likelihood_new_log:", p_y_likelihood_new_log)
            print("p_y_likelihood_current_log:", p_y_likelihood_current_log)
            p_x_new_log = p_x_prior_new_log + p_y_likelihood_new_log
            p_x_current_log = p_x_prior_current_log + p_y_likelihood_current_log

            print("p_x_new_log:", p_x_new_log)
            print("p_x_current_log:", p_x_current_log)
            accept_ratio_init = jnp.exp(p_x_new_log - p_x_current_log)
            accept_ratio = jnp.minimum(accept_ratio_init, 1)
            print("-----------------------------------------------------------------------")
            print(f"Iteration {i}, Variable {j}")
            print(f"Current sample: {xvague_sample_current}")
            print(f"New sample: {x_new}")
            print(f"Acceptance ratio: {accept_ratio}")
            print("-----------------------------------------------------------------------")
            print("\n")

            rng_key, key_uniform = random.split(rng_key)
            if random.uniform(key_uniform) < accept_ratio:
                xvague_sample_current = x_new
                current_log_prob = p_x_prior_new_log
                acceptance_count += 1
                acceptance_flags.append(1)
            else:
                acceptance_flags.append(0)

        if i >= num_warmup:
            xvague_sample_list = jnp.vstack((xvague_sample_list, xvague_sample_current))

        if i % 100 == 0:
            print(
                f"Iteration {i}: Current sample = {xvague_sample_current}, new prior probability = {current_log_prob}, new likelihood = {p_y_likelihood_new_log}, new log prob = {p_x_new_log}")

        if (i + 1) % adjustment_interval == 0 and i < num_warmup:
            recent_acceptance_rate = jnp.mean(jnp.array(acceptance_flags[-adjustment_interval:]))
            if recent_acceptance_rate < target_acceptance_rate - tolerance:
                assumption_sigma *= 0.9  # Decrease step size slightly
            elif recent_acceptance_rate > target_acceptance_rate + tolerance:
                assumption_sigma *= 1.1  # Increase step size slightly

    acceptance_rate = jnp.mean(jnp.array(acceptance_flags))
    print(f"#######Acceptance rate: {acceptance_rate}########")

    return xvague_sample_list


def single_component_metropolis_hasting_rd(rng_key, max_samples, assumption_sigma, prior_mean, prior_cov, init, Xfz,
                                           Xfg, Y, k):
    prior_mean_flat = jnp.ravel(prior_mean)
    prior_cov_flat = jnp.kron(prior_cov, jnp.eye(prior_mean.shape[0]))  # shape (2*num_vague, 2*num_vague)

    num_vague = prior_mean.shape[0]
    num_x_t_noise = num_vague

    xvague_sample_current = prior_mean_flat
    xvague_sample_list = jnp.empty((0, prior_mean_flat.shape[0]), dtype=prior_mean_flat.dtype)

    current_log_prob = dist.MultivariateNormal(prior_mean_flat, covariance_matrix=prior_cov_flat).log_prob(
        xvague_sample_current)
    print("current_log_prob:", current_log_prob)
    num_warmup = int(max_samples * k)
    print(f"Setting {k} of max_samples as warm-up")
    acceptance_count = 0
    print("num_warmup:", num_warmup)
    acceptance_flags = []

    for i in range(max_samples - 1):
        for j in range(num_x_t_noise * 2):
            rng_key, key_new = random.split(rng_key)
            step_size = assumption_sigma

            if j % 2 == 0:
                new_value = random.normal(key_new, shape=()) * step_size + xvague_sample_current[j]
                new_value = jnp.maximum(jnp.minimum(1, new_value), -1)
            else:
                new_value = random.normal(key_new, shape=()) * step_size + xvague_sample_current[j]
                new_value = jnp.maximum(jnp.minimum(1, new_value), 0)

            x_new = xvague_sample_current.at[j].set(new_value)
            x_new_reshape = x_new.reshape(prior_mean.shape)
            xvague_sample_current_reshape = xvague_sample_current.reshape(prior_mean.shape)

            p_x_prior_new_log = dist.MultivariateNormal(prior_mean_flat, covariance_matrix=prior_cov_flat).log_prob(
                x_new)
            p_x_prior_current_log = dist.MultivariateNormal(prior_mean_flat, covariance_matrix=prior_cov_flat).log_prob(
                xvague_sample_current)
            cov_current = compute_K_rd(init, xvague_sample_current_reshape, Xfz, Xfg)
            cov_new = compute_K_rd(init, x_new_reshape, Xfz, Xfg)

            if jnp.isnan(cov_new).any() or jnp.isnan(cov_current).any():
                print("Covariance matrix has NaN values!")

            Y_ravel = Y.ravel()
            p_y_likelihood_new_log = dist.MultivariateNormal(jnp.zeros(Y_ravel.shape),
                                                             covariance_matrix=cov_new).log_prob(Y_ravel)
            p_y_likelihood_current_log = dist.MultivariateNormal(jnp.zeros(Y_ravel.shape),
                                                                 covariance_matrix=cov_current).log_prob(Y_ravel)
            print("p_y_likelihood_new_log:", p_y_likelihood_new_log)
            print("p_y_likelihood_current_log:", p_y_likelihood_current_log)

            p_x_new_log = p_x_prior_new_log + p_y_likelihood_new_log
            p_x_current_log = p_x_prior_current_log + p_y_likelihood_current_log

            accept_ratio_init = jnp.exp(p_x_new_log - p_x_current_log)
            accept_ratio = jnp.minimum(accept_ratio_init, 1)
            print("-----------------------------------------------------------------------")
            print(f"Iteration {i}, Variable {j}")
            print(f"Current sample: {xvague_sample_current}")
            print(f"New sample: {x_new}")
            print(f"Acceptance ratio: {accept_ratio}")
            print("-----------------------------------------------------------------------")
            print("\n")

            rng_key, key_uniform = random.split(rng_key)
            if random.uniform(key_uniform) < accept_ratio:
                xvague_sample_current = x_new
                current_log_prob = p_x_prior_new_log
                acceptance_count += 1
                acceptance_flags.append(1)
            else:
                acceptance_flags.append(0)

        if i >= num_warmup:
            xvague_sample_list = jnp.vstack((xvague_sample_list, xvague_sample_current))

        if i % 100 == 0:
            print(
                f"Iteration {i}: Current sample = {xvague_sample_current}, new prior probability = {current_log_prob}, new likelihood = {p_y_likelihood_new_log}, new log prob = {p_x_new_log}")

        target_acceptance_rate = 0.3
        tolerance = 0.05
        adjustment_interval = 100

        if (i + 1) % adjustment_interval == 0 and i < num_warmup:
            recent_acceptance_rate = jnp.mean(jnp.array(acceptance_flags[-adjustment_interval:]))
            if recent_acceptance_rate < target_acceptance_rate - tolerance:
                assumption_sigma *= 0.9  # Decrease step size slightly
            elif recent_acceptance_rate > target_acceptance_rate + tolerance:
                assumption_sigma *= 1.1  # Increase step size slightly

    acceptance_rate = jnp.mean(jnp.array(acceptance_flags))
    print(f"#######Acceptance rate: {acceptance_rate}########")
    return xvague_sample_list


def metropolis_hasting(rng_key, max_samples, assumption_sigma, prior_mean, prior_cov, init, Xfz, Xfg, Y, k):
    prior_mean_flat = jnp.ravel(prior_mean)
    prior_cov_flat = jnp.kron(prior_cov, jnp.eye(prior_mean.shape[0]))  # shape (2*num_vague, 2*num_vague)

    num_vague = prior_mean.shape[0]
    xvague_sample_current = prior_mean_flat
    xvague_sample_list = jnp.empty((0, prior_mean_flat.shape[0]), dtype=prior_mean_flat.dtype)

    current_log_prob = dist.MultivariateNormal(prior_mean_flat, covariance_matrix=prior_cov_flat).log_prob(
        xvague_sample_current)

    num_warmup = int(max_samples * k)
    print(f"Setting {k} of max_samples as warm-up")
    acceptance_count = 0
    print("num_warmup:", num_warmup)
    x_new = prior_mean_flat
    acceptance_flags = []
    for i in range(max_samples - 1):
        rng_key, key_new = random.split(rng_key)

        step_size = assumption_sigma

        x_new = random.multivariate_normal(key_new, mean=xvague_sample_current, cov=jnp.eye(num_vague * 2) * step_size)

        x_new = jnp.maximum(jnp.minimum(1, x_new), 0)  # Non-negative

        x_new_reshape = x_new.reshape(prior_mean.shape)
        xvague_sample_current_reshape = xvague_sample_current.reshape(prior_mean.shape)

        p_x_prior_new_log = dist.MultivariateNormal(prior_mean_flat, covariance_matrix=prior_cov_flat).log_prob(x_new)
        p_x_prior_current_log = dist.MultivariateNormal(prior_mean_flat, covariance_matrix=prior_cov_flat).log_prob(
            xvague_sample_current)
        cov_current = compute_K(init, xvague_sample_current_reshape, Xfz, Xfg)
        cov_new = compute_K(init, x_new_reshape, Xfz, Xfg)

        Y_ravel = Y.ravel()
        p_y_likelihood_new_log = dist.MultivariateNormal(jnp.zeros(Y_ravel.shape), covariance_matrix=cov_new).log_prob(
            Y_ravel)
        p_y_likelihood_current_log = dist.MultivariateNormal(jnp.zeros(Y_ravel.shape),
                                                             covariance_matrix=cov_current).log_prob(Y_ravel)

        p_x_new_log = p_x_prior_new_log + p_y_likelihood_new_log
        p_x_current_log = p_x_prior_current_log + p_y_likelihood_current_log

        accept_ratio_init = jnp.exp(p_x_new_log - p_x_current_log)
        accept_ratio = jnp.minimum(accept_ratio_init, 1)
        print("new sample is :", x_new)
        print(f"ratio for {i}: {accept_ratio}")

        rng_key, key_uniform = random.split(rng_key)
        if random.uniform(key_uniform) < accept_ratio:
            xvague_sample_current = x_new
            current_log_prob = p_x_prior_new_log
            acceptance_count += 1
            acceptance_flags.append(1)
        else:
            acceptance_flags.append(0)
        if i >= num_warmup:
            xvague_sample_list = jnp.vstack((xvague_sample_list, xvague_sample_current))
        if i % 100 == 0:
            print(
                f"Iteration {i}: Current sample = {xvague_sample_current}, new prior probability = {current_log_prob}, new likelihood = {p_y_likelihood_new_log}, new log prob = {p_x_new_log}")

    acceptance_rate = jnp.mean(jnp.array(acceptance_flags))
    print(f"#######Acceptance rate: {acceptance_rate}########")
    print("cov_current:", cov_current)
    print("cov_new:", cov_new)
    return xvague_sample_list


def model(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params):
    prior_mean_flat = jnp.ravel(prior_mean)
    prior_cov_flat = jnp.kron(prior_cov, jnp.eye(prior_mean.shape[0]))

    z_flat = numpyro.sample('z_uncertain_flat',
                            dist.MultivariateNormal(prior_mean_flat, covariance_matrix=prior_cov_flat))
    z_uncertain = z_flat.reshape(prior_mean.shape)

    K = compute_K(init_params, z_uncertain, Xfz, Xfg)
    K = jnp.array(K)
    numpyro.sample('z_obs', dist.MultivariateNormal(jnp.zeros(K.shape[0]), covariance_matrix=K), obs=Y_data)


def run_mcmc(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params, num_samples=5, num_warmup=10):
    rng_key = random.PRNGKey(4)
    kernel = BarkerMH(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(rng_key, Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params)
    mcmc.print_summary()
    return mcmc


def prior_function(x, prior_mean, covariance_matrix):
    return -0.5 * (x - prior_mean).T @ np.linalg.inv(covariance_matrix) @ (x - prior_mean)


def likelihood(y, covariance_matrix):
    return -0.5 * y.T @ np.linalg.inv(covariance_matrix) @ y


def Metropolis_Hasting(max_samples, assumption_variance, Xvague_prior_mean, Xvague_prior_var, init, Xfz, Xfg, Y):

    num_uncertain_points = Xvague_prior_mean.shape[0]
    Xvague_prior_mean = np.vstack((Xvague_prior_mean[:, 0:1], Xvague_prior_mean[:, 1:2]))

    num_vague = Xvague_prior_mean.shape[0]
    xvague_sample_current = Xvague_prior_mean
    num_samples = 1

    xvague_sample_list = np.empty((num_vague, 0))  ### List of samples
    xvague_sample_list = np.hstack((xvague_sample_list, xvague_sample_current))

    for i in range(1):

        x_new = np.random.multivariate_normal(np.squeeze(xvague_sample_current),
                                              jnp.eye(num_vague) * assumption_variance, 1).T
        x_new[x_new < 0] = 0

        prior_function_upper = prior_function(x_new, Xvague_prior_mean, Xvague_prior_var)
        prior_function_lower = prior_function(xvague_sample_current, Xvague_prior_mean, Xvague_prior_var)

        covariance_matrix_upper = compute_K(init, x_new.reshape(-1, 2, order='F'), Xfz, Xfg)
        covariance_matrix_lower = compute_K(init, xvague_sample_current.reshape(-1, 2, order='F'), Xfz, Xfg)

        likelihood_upper = likelihood(Y, covariance_matrix_upper)
        likelihood_lower = likelihood(Y, covariance_matrix_lower)

        print('New:', np.squeeze(x_new), '; Prior:', prior_function_upper, 'LH:', likelihood_upper)
        print('Current:', np.squeeze(xvague_sample_current), '; Prior:', prior_function_lower, 'LH:', likelihood_lower)
        print('Det ratio:', np.sqrt(np.linalg.det(covariance_matrix_lower) / np.linalg.det(covariance_matrix_upper)))

        accept_ratio = np.sqrt(np.linalg.det(covariance_matrix_lower) / np.linalg.det(covariance_matrix_upper)) * \
                       np.exp(prior_function_upper + likelihood_upper - prior_function_lower - likelihood_lower)
        check_sample = np.squeeze(np.random.uniform(0, 1, 1))

        if check_sample <= accept_ratio:
            xvague_sample_current = x_new
            xvague_sample_list = np.hstack((xvague_sample_list, xvague_sample_current))

            print('Accept ratio: ', accept_ratio, '; accept sample: ', check_sample)
            num_samples = num_samples + 1
        else:

            pass
    return xvague_sample_list


def neg_log_posterior(params_flat, Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params):
    prior_mean_flat = jnp.ravel(prior_mean)
    prior_cov_flat = jnp.kron(prior_cov, jnp.eye(prior_mean.shape[0]))

    neg_log_prior = -dist.MultivariateNormal(prior_mean_flat, prior_cov_flat).log_prob(params_flat)

    z_uncertain = params_flat.reshape(prior_mean.shape)

    K = compute_K(init_params, z_uncertain, Xfz, Xfg)
    neg_log_likelihood = -dist.MultivariateNormal(jnp.zeros(K.shape[0]), K).log_prob(Y_data)

    return neg_log_prior + neg_log_likelihood


"""
def model(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params):
    prior_mean = jnp.array(prior_mean)
    prior_cov = jnp.array(prior_cov)
    z_uncertain_list = []
    print("prior_mean shape:", prior_mean.shape[0])
    for num_sample in range(prior_mean.shape[0]):
        z_uncertain = numpyro.sample('z_uncertain' + str(num_sample),
                                     dist.MultivariateNormal(prior_mean[num_sample, :], covariance_matrix=prior_cov))
        z_uncertain_list.append(z_uncertain)

    z_uncertain = jnp.stack(z_uncertain_list)
    print("#############################################")
    print("z_uncertain_list shape", z_uncertain.shape)
    print("z_uncertain_list", z_uncertain_list)
    print("#############################################")
    K = compute_K(init_params, z_uncertain, Xfz, Xfg)
    print("#############################################")
    print("#############################################")
    print("K shape:", K.shape)
    print("K:", K)
    print("#############################################")
    K = jnp.array(K)
    numpyro.sample('z_obs', dist.MultivariateNormal(jnp.zeros(K.shape[0]), covariance_matrix=K), obs=jnp.array(Y_data))

def run_mcmc(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params, num_samples=5, num_warmup=10):
    rng_key = random.PRNGKey(42)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params)
    mcmc.print_summary()
    return mcmc
"""
