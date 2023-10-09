# %%

import jax.numpy as jnp
import numpy
import numpy as np
import pymc as pm
from include.heat2d import heat_equation_kuu_noise, heat_equation_kuf, heat_equation_kfu, heat_equation_kff


def compute_K(init, z_prior, Xfz, Xfg):
    Xuz = z_prior
    params = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    zz_uu = heat_equation_kuu_noise(Xuz, Xuz, params)
    zz_uf = heat_equation_kuf(Xuz, Xfz, params)
    zg_uf = heat_equation_kfu(Xuz, Xfg, params)
    zz_fu = heat_equation_kfu(Xfz, Xuz, params)
    zz_ff = heat_equation_kff(Xfz, Xfz, params)
    zg_ff = heat_equation_kff(Xfz, Xfg, params)
    gz_fu = heat_equation_kfu(Xfg, Xuz, params)
    gz_ff = heat_equation_kff(Xfg, Xfz, params)
    gg_ff = heat_equation_kff(Xfg, Xfg, params)
    K = jnp.block([[zz_uu, zz_uf, zg_uf], [zz_fu, zz_ff, zg_ff], [gz_fu, gz_ff, gg_ff]])
    return K

def posterior_inference_mcmc(prior_mean, prior_std, init_params, Xfz, Xfg, Y_data):

    basic_model = pm.Model()
    prior_mean = numpy.array(prior_mean)
    # prior_var = numpy.array(prior_var)


    with basic_model:
        z_uncertain_x_list = []
        z_uncertain_t_list = []
        for num_sample in range(prior_mean.shape[0]):
            z_uncertain_x_list.append(pm.Normal('x_uncertain'+str(num_sample), mu=prior_mean[num_sample, 0],
                                                sigma=prior_std))
            z_uncertain_t_list.append(pm.Normal('t_uncertain'+str(num_sample), mu=prior_mean[num_sample, 1],
                                                sigma=prior_std))

        z_uncertain_x_list = numpy.array(z_uncertain_x_list)
        z_uncertain_t_list = numpy.array(z_uncertain_t_list)
        z_uncertain_list = numpy.vstack((np.expand_dims(z_uncertain_x_list, axis=1),
                                         np.expand_dims(z_uncertain_t_list, axis=1)))
        # z_uncertain = jnp.array(z_uncertain_samples)
        # print("-------------------z_uncertain list:--------------------------", z_uncertain_list)
        # z_uncertain = numpy.asarray(z_uncertain_list)
        # z_uncertain_1 = numpy.asarray(z_uncertain_list[0])
        # print("-------------------1:--------------------------", z_uncertain_1)
        # # z_uncertain = jnp.array(z_uncertain_numpy)
        # # print("-------------------z_uncertain shape:--------------------------", z_uncertain_list.shape)
        # # print("-------------------z_uncertain type:--------------------------", z_uncertain_list.dtype)
        # print("-------------------z_uncertain:--------------------------", z_uncertain_list)
        # K = compute_K(init_params, z_uncertain, Xfz, Xfg)
        K = np.identity(z_uncertain_list.shape[0]+Xfz.shape[0]+Xfg.shape[0])
        K = numpy.array(K)
        k_shape = K.shape[0]
        z_obs = pm.MvNormal('z_obs', mu=numpy.zeros(k_shape), cov=K, observed=numpy.array(Y_data))
        trace_final = pm.sample(500, return_inferencedata=False)

    return trace_final



