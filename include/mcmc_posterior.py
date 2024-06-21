# %%
import numpy
import numpy as np
import pymc as pm
from .heat2d import heat_equation_kuu_noise, heat_equation_kuf, heat_equation_kfu, heat_equation_kff, \
    heat_equation_kff_noise, heat_equation_kuu, heat_equation_kuu_noise2

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
import jax.numpy as jnp
from jax import random
from collections import namedtuple

def compute_K(init, z_prior, Xfz, Xfg):
    Xuz = z_prior
    params = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    zz_uu = heat_equation_kuu(Xuz, Xuz, params)
    zz_uf = heat_equation_kuu(Xuz, Xfz, params)
    zg_uf = heat_equation_kuf(Xuz, Xfg, params)
    zz_fu = heat_equation_kuu(Xfz, Xuz, params)
    zz_ff = heat_equation_kuu(Xfz, Xfz, params)
    zg_ff = heat_equation_kuf(Xfz, Xfg, params)
    gz_fu = heat_equation_kfu(Xfg, Xuz, params)
    gz_ff = heat_equation_kfu(Xfg, Xfz, params)
    gg_ff = heat_equation_kff(Xfg, Xfg, params)
    K = jnp.block([[zz_uu, zz_uf, zg_uf], [zz_fu, zz_ff, zg_ff], [gz_fu, gz_ff, gg_ff]])
    return K
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

# MHState = namedtuple("MHState", ["u", "rng_key"])
# class MetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
#     sample_field = "u"
#
#     def __init__(self, potential_fn, step_size=0.1):
#         self.potential_fn = potential_fn
#         self.step_size = step_size
#
#     def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
#         return MHState(init_params, rng_key)
#
#     def sample(self, state, model_args, model_kwargs):
#         u, rng_key = state
#         rng_key, key_proposal, key_accept = random.split(rng_key, 3)
#         u_proposal = dist.Normal(u, self.step_size).sample(key_proposal)
#         accept_prob = jnp.exp(self.potential_fn(u) - self.potential_fn(u_proposal))
#         u_new = jnp.where(dist.Uniform().sample(key_accept) < accept_prob, u_proposal, u)
#         return MHState(u_new, rng_key)


def run_mcmc(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params, num_samples=5, num_warmup=10):
    rng_key = random.PRNGKey(42)
    # kernel = MetropolisHastings(model) # No U-Turn Sampler
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params)
    mcmc.print_summary()
    return mcmc
"""


# def model(Xfz, Xfg, Y_data, prior_mean, prior_cov,  init_params):
#     # 先验分布
#     z_uncertain = numpyro.sample('z_uncertain', dist.MultivariateNormal(prior_mean, covariance_matrix=prior_cov))
#
#     # 似然分布
#     K = compute_K(init_params, z_uncertain, Xfz, Xfg)
#     K = jnp.array(K)
#     numpyro.sample('z_obs', dist.MultivariateNormal(jnp.zeros(K.shape[0]), covariance_matrix=K), obs=Y_data)


# def model(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params):
#     z_uncertain = []
#     for j in range(prior_mean.shape[1]):  #
#         z = numpyro.sample(f'z_uncertain_{j}', dist.MultivariateNormal(prior_mean[:, j], covariance_matrix=prior_cov[j]))
#         z_uncertain.append(z)
#     z_uncertain = jnp.stack(z_uncertain, axis=1)
#
#     #
#     K = compute_K(init_params, z_uncertain, Xfz, Xfg)
#     K = jnp.array(K)
#     numpyro.sample('z_obs', dist.MultivariateNormal(jnp.zeros(K.shape[0]), covariance_matrix=K), obs=Y_data)


def model(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params):
    prior_mean_flat = jnp.ravel(prior_mean)
    prior_cov_flat = jnp.kron(prior_cov, jnp.eye(prior_mean.shape[0]))

    z_flat = numpyro.sample('z_uncertain_flat', dist.MultivariateNormal(prior_mean_flat, covariance_matrix=prior_cov_flat))
    z_uncertain = z_flat.reshape(prior_mean.shape)

    K = compute_K(init_params, z_uncertain, Xfz, Xfg)
    K = jnp.array(K)
    numpyro.sample('z_obs', dist.MultivariateNormal(jnp.zeros(K.shape[0]), covariance_matrix=K), obs=Y_data)


def run_mcmc(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params, num_samples=5, num_warmup=10):
    rng_key = random.PRNGKey(4)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params)
    mcmc.print_summary()
    return mcmc


def neg_log_posterior(params_flat, Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params):
    prior_mean_flat = jnp.ravel(prior_mean)
    prior_cov_flat = jnp.kron(prior_cov, jnp.eye(prior_mean.shape[0]))

    # 计算负对数先验概率
    neg_log_prior = -dist.MultivariateNormal(prior_mean_flat, prior_cov_flat).log_prob(params_flat)

    # 恢复形状
    z_uncertain = params_flat.reshape(prior_mean.shape)

    # 计算负对数似然
    K = compute_K(init_params, z_uncertain, Xfz, Xfg)
    neg_log_likelihood = -dist.MultivariateNormal(jnp.zeros(K.shape[0]), K).log_prob(Y_data)

    return neg_log_prior + neg_log_likelihood
#
# def prior_function(x, prior_mean, covariance_matrix):
#     return (x - prior_mean).T @ np.linalg.inv(covariance_matrix) @ (x - prior_mean)
#
# def likelihood(y, covariance_matrix):
#     # x.shape = (num_dim=40,)
#     print("y shape:", y.shape)
#     print("covariance_matrix shape:", covariance_matrix.shape)
#     return y.T@np.linalg.inv(covariance_matrix)@y

#
# def Metropolis_Hasting(max_samples, assumption_variance, Xvague_prior_mean, Xvague_prior_var, init, Xfz, Xfg, Y):
#     Xvague_prior_mean = np.vstack((Xvague_prior_mean[:, 0:1], Xvague_prior_mean[:, 1:2]))
#     ### Input: num_vague,xvauge_prior_mean, xvague_prior_var
#     num_vague = Xvague_prior_mean.shape[0]
#     xvague_sample_current = Xvague_prior_mean
#     num_samples = 1
#
#     xvague_sample_list = np.empty((num_vague, 0))  ### List of samples
#     xvague_sample_list = np.hstack((xvague_sample_list, xvague_sample_current))
#
#
#     while num_samples < max_samples:
#
#         x_new = np.random.multivariate_normal(np.squeeze(xvague_sample_current), jnp.eye(num_vague)*assumption_variance,
#                                               1).T
#         x_new[x_new < 0] = 0
#
#         prior_function_upper = prior_function(x_new, Xvague_prior_mean, Xvague_prior_var)
#         prior_function_lower = prior_function(xvague_sample_current, Xvague_prior_mean, Xvague_prior_var)
#
#         covariance_matrix_upper = compute_K(init, x_new, Xfz, Xfg)
#         covariance_matrix_lower = compute_K(init, xvague_sample_current, Xfz, Xfg)
#
#         likelihood_upper = likelihood(Y, covariance_matrix_upper)
#         likelihood_lower = likelihood(Y, covariance_matrix_lower)
#
#         accept_ratio = np.exp(prior_function_upper+likelihood_upper-prior_function_lower-likelihood_lower)
#         check_sample = np.squeeze(np.random.uniform(0, 1, 1))
#
#         if check_sample <= accept_ratio:
#             xvague_sample_current = x_new
#             xvague_sample_list = np.hstack((xvague_sample_list, xvague_sample_current))
#             # print('Accept ratio: ',accept_ratio,'; Xnew: ',x_new,'; Accept')
#         else:
#             pass
#     return xvague_sample_list


# def posterior_inference_mcmc(prior_mean,prior_cov, init_params, Xfz, Xfg, Y_data):
#
#     basic_model = pm.Model()
#     prior_mean = numpy.array(prior_mean)
#     prior_cov = numpy.array(prior_cov)
#
#
#     with basic_model:
#         z_uncertain_list = []
#         for num_sample in range(prior_mean.shape[0]):
#             z_uncertain_list.append(pm.MvNormal('z_uncertain'+str(num_sample),mu=prior_mean[num_sample,:],cov=prior_cov))
#         z_uncertain = numpy.array(z_uncertain_list)
#         print(z_uncertain)
#         # print("-------------------z_uncertian shape:--------------------------", z_uncertain_list.shape)
#
#         K = compute_K(init_params, z_uncertain, Xfz, Xfg)
#         K = numpy.array(K)
#         k_shape = K.shape[0]
#         z_obs = pm.MvNormal('z_obs',mu=numpy.zeros(k_shape),cov=K,observed=numpy.array(Y_data))
#         step = pm.Metropolis()
#         trace = pm.sample(500,step=step,return_inferencedata=False)
#
#     return trace

# def posterior_numpyro(prior_mean,prior_cov, init_params, Xfz, Xfg, Y_data):
#     # basic_model = pm.Model()
#     prior_mean = numpy.array(prior_mean)
#     prior_cov = numpy.array(prior_cov)
#
#     print(Xfz.shape,Xfg.shape,Y_data.shape)
#
#     def basic_model(X,y):
#         z_uncertain_list = []
#         for num_sample in range(prior_mean.shape[0]):
#             z_uncertain_list.append(numpyro.sample('z_uncertain'+str(num_sample),dist.MultivariateNormal(prior_mean[num_sample,:],prior_cov)))
#             # z_uncertain_list.append(pm.MvNormal('z_uncertain'+str(num_sample),mu=prior_mean[num_sample,:],cov=prior_cov))
#         print(z_uncertain_list)
#
#         z_uncertain = numpy.array(z_uncertain_list)
#         print(z_uncertain)
#         # print("-------------------z_uncertian shape:--------------------------", z_uncertain_list.shape)
#         K = compute_K(init_params, z_uncertain, Xfz, Xfg)
#         K = numpy.array(K)
#         k_shape = K.shape[0]
#         z_obs = numpyro.sample('obs',dist.MultivariateNormal(numpy.zeros(k_shape),K),obs=y)
#         # z_obs = pm.MvNormal('z_obs',mu=numpy.zeros(k_shape),cov=K,observed=numpy.array(Y_data))
#         # step = pm.Metropolis()
#         # trace = pm.sample(500,step=step,return_inferencedata=False)
#
#     return 0
#     # def model(X, y):
#     #     beta = numpyro.sample("beta", dist.Normal(0, 1).expand([3]))
#     #     numpyro.sample("obs", dist.Normal(X @ beta, 1), obs=y)
#
#     # mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
#     # # See https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
#     # sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
#     # X_shard = jax.device_put(X, sharding.reshape(8, 1))
#     # y_shard = jax.device_put(y, sharding.reshape(8))
#     # mcmc.run(jax.random.PRNGKey(0), X_shard, y_shard)




# trace = run_mcmc(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params)
