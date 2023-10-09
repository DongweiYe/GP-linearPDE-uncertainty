# %%

import jax.numpy as jnp
import numpy
import numpy as np
# import pymc as pm
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

def prior_function(x, prior_mean, covariance_matrix):
    return (x - prior_mean).T @ np.linalg.inv(covariance_matrix) @ (x - prior_mean)

def likelihood(y, covariance_matrix):
    # x.shape = (num_dim=40,)
    print("y shape:", y.shape)
    print("covariance_matrix shape:", covariance_matrix.shape)
    return y.T@np.linalg.inv(covariance_matrix)@y


def Metropolis_Hasting(max_samples, assumption_variance, Xvague_prior_mean, Xvague_prior_var, init, Xfz, Xfg, Y):
    Xvague_prior_mean = np.vstack((Xvague_prior_mean[:, 0:1], Xvague_prior_mean[:, 1:2]))
    ### Input: num_vague,xvauge_prior_mean, xvague_prior_var
    num_vague = Xvague_prior_mean.shape[0]
    xvague_sample_current = Xvague_prior_mean
    num_samples = 1

    xvague_sample_list = np.empty((num_vague, 0))  ### List of samples
    xvague_sample_list = np.hstack((xvague_sample_list, xvague_sample_current))


    while num_samples < max_samples:

        x_new = np.random.multivariate_normal(np.squeeze(xvague_sample_current), jnp.eye(num_vague)*assumption_variance,
                                              1).T
        x_new[x_new < 0] = 0

        prior_function_upper = prior_function(x_new, Xvague_prior_mean, Xvague_prior_var)
        prior_function_lower = prior_function(xvague_sample_current, Xvague_prior_mean, Xvague_prior_var)

        covariance_matrix_upper = compute_K(init, x_new, Xfz, Xfg) 
        covariance_matrix_lower = compute_K(init, xvague_sample_current, Xfz, Xfg)

        likelihood_upper = likelihood(Y, covariance_matrix_upper)
        likelihood_lower = likelihood(Y, covariance_matrix_lower)

        accept_ratio = np.exp(prior_function_upper+likelihood_upper-prior_function_lower-likelihood_lower)
        check_sample = np.squeeze(np.random.uniform(0, 1, 1))

        if check_sample <= accept_ratio:
            xvague_sample_current = x_new
            xvague_sample_list = np.hstack((xvague_sample_list, xvague_sample_current))
            # print('Accept ratio: ',accept_ratio,'; Xnew: ',x_new,'; Accept')
        else:
            pass
    return xvague_sample_list
