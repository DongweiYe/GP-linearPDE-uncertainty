# %%

import jax.numpy as jnp
import numpy
import pymc as pm
from .heat2d import heat_equation_kuu_noise, heat_equation_kuf, heat_equation_kfu, heat_equation_kff


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

def posterior_inference_mcmc(prior_mean,prior_cov, init_params, Xfz, Xfg, Y_data):

    basic_model = pm.Model()
    prior_mean = numpy.array(prior_mean)
    prior_cov = numpy.array(prior_cov)


    with basic_model:
        z_uncertain_list = []
        for num_sample in range(prior_mean.shape[0]):
            z_uncertain_list.append(pm.MvNormal('z_uncertain'+str(num_sample),mu=prior_mean[num_sample,:],cov=prior_cov))
        z_uncertain = numpy.array(z_uncertain_list)
        print("-------------------z_uncertian shape:--------------------------", z_uncertain_list.shape)
        K = compute_K(init_params, z_uncertain, Xfz, Xfg)
        K = numpy.array(K)
        k_shape = K.shape[0]
        z_obs = pm.MvNormal('z_obs',mu=numpy.zeros(k_shape),cov=K,observed=numpy.array(Y_data))
        trace = pm.sample(500,return_inferencedata=False)

    return trace


