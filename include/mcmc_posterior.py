# %%
import numpy
import numpy as np
import pymc as pm
from .heat2d import heat_equation_kuu_noise, heat_equation_kuf, heat_equation_kfu, heat_equation_kff, heat_equation_kuu ,heat_equation_kff_noise, heat_equation_kuu_noise_minor

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, BarkerMH
import jax.numpy as jnp
from jax import random
from collections import namedtuple


### Covairance matrix
def compute_K_local(init, z_prior, Xfz):
    Xuz = z_prior
    params = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    # print(params)
    zz_uu = heat_equation_kuu_noise(Xuz, Xuz, params)  ### K11
    gg_ff = heat_equation_kff(Xfz, Xfz, params) ### K33
    zg_uf = heat_equation_kuf(Xuz, Xfz, params) ### K13
    gz_fu = heat_equation_kfu(Xfz, Xuz, params) ### K31

    
    K = jnp.block([[zz_uu, zg_uf], [gz_fu, gg_ff]])
    return K

def compute_K(init, z_prior, Xfz, Xfg):
    Xuz = z_prior
    params = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    # print(params)
    zz_uu = heat_equation_kuu_noise(Xuz, Xuz, params)  ### K11
    zz_ff = heat_equation_kuu_noise_minor(Xfz, Xfz, params)  ### K22
    gg_ff = heat_equation_kff_noise(Xfg, Xfg, params) ### K33

    # zz_uu = heat_equation_kuu(Xuz, Xuz, params) ### K11
    # zz_ff = heat_equation_kuu(Xfz, Xfz, params) ### K22
    # gg_ff = heat_equation_kff(Xfg, Xfg, params) ### K33

    zz_uf = heat_equation_kuu(Xuz, Xfz, params) ### K12
    zg_uf = heat_equation_kuf(Xuz, Xfg, params) ### K13

    zz_fu = heat_equation_kuu(Xfz, Xuz, params) ### K21
    zg_ff = heat_equation_kuf(Xfz, Xfg, params) ### K23

    gz_fu = heat_equation_kfu(Xfg, Xuz, params) ### K31
    gz_ff = heat_equation_kfu(Xfg, Xfz, params) ### K32
    
    K = jnp.block([[zz_uu, zz_uf, zg_uf], [zz_fu, zz_ff, zg_ff], [gz_fu, gz_ff, gg_ff]])
    return K


def model(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params):
    prior_mean = jnp.array(prior_mean)
    prior_cov = jnp.array(prior_cov)
    z_uncertain_list = []

    for num_uncertain_points in range(prior_mean.shape[0]):
        z_uncertain = numpyro.sample('z_uncertain' + str(num_uncertain_points),
                                     dist.MultivariateNormal(prior_mean[num_uncertain_points,:], covariance_matrix=prior_cov))
        z_uncertain_list.append(z_uncertain)

    z_uncertain = jnp.stack(z_uncertain_list)
    K = compute_K(init_params, z_uncertain, Xfz, Xfg)
    K = jnp.array(K)
    numpyro.sample('z_obs', dist.MultivariateNormal(jnp.zeros(K.shape[0]), covariance_matrix=K), obs=jnp.array(Y_data))


def run_mcmc(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params, num_samples=5, num_warmup=10):
    print('Xfz (boundary conditions):',Xfz.shape)
    print('Xfg (external source):',Xfg.shape)
    print('Y (not sure):',Y_data.shape)
    print('Prior_mean (Xuz):',prior_mean.shape)
    print('Prior_cova (Xuz):',prior_cov.shape)
    rng_key = random.PRNGKey(0)
    # kernel = MetropolisHastings(model)
    kernel = BarkerMH(model)
    # kernel = NUTS(model) # No U-Turn Sampler
    # kernel = AIES(model)
    # kernel = HMC(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params)
    mcmc.print_summary()
    return mcmc.get_samples()



def prior_function(x, prior_mean, covariance_matrix):
    return -0.5*(x - prior_mean).T @ np.linalg.inv(covariance_matrix) @ (x - prior_mean)

def likelihood(y, covariance_matrix):
    # x.shape = (num_dim=40,)
    # print("y shape:", y.shape)
    # print("covariance_matrix shape:", covariance_matrix.shape)
    return -0.5*y.T@np.linalg.inv(covariance_matrix)@y

def Metropolis_Hasting(max_samples, assumption_variance, Xvague_prior_mean, Xvague_prior_var, init, Xfz, Xfg, Y):
    ### Prior mean is now reshape into a 1D vector t is attached to the end of x.
    num_uncertain_points = Xvague_prior_mean.shape[0]
    Xvague_prior_mean = np.vstack((Xvague_prior_mean[:, 0:1], Xvague_prior_mean[:, 1:2]))
    
    ### Input: num_vague,xvauge_prior_mean, xvague_prior_var
    num_vague = Xvague_prior_mean.shape[0]
    xvague_sample_current = Xvague_prior_mean
    num_samples = 1

    xvague_sample_list = np.empty((num_vague, 0))  ### List of samples
    xvague_sample_list = np.hstack((xvague_sample_list, xvague_sample_current))

    for i in range(1):
    # while num_samples < max_samples:
        
        ### Note that x_new and xvague_sample_current are in shape (num_uncertianty_points*2,1) as vector
        ### But when computing matrix K, it should be reshaped into (num_uncertianty_points,2)
        ### which reshape back to data form
        
        x_new = np.random.multivariate_normal(np.squeeze(xvague_sample_current), jnp.eye(num_vague)*assumption_variance,1).T
        x_new[x_new < 0] = 0

        # x_new = np.array([[0.5],[0.5]])


        ### A huge multivariate Gaussian distribution for all the uncertainty priors 
        prior_function_upper = prior_function(x_new, Xvague_prior_mean, Xvague_prior_var)
        prior_function_lower = prior_function(xvague_sample_current, Xvague_prior_mean, Xvague_prior_var)

        covariance_matrix_upper = compute_K(init, x_new.reshape(-1,2,order='F'), Xfz, Xfg) 
        covariance_matrix_lower = compute_K(init, xvague_sample_current.reshape(-1,2,order='F'), Xfz, Xfg)

        likelihood_upper = likelihood(Y, covariance_matrix_upper)
        likelihood_lower = likelihood(Y, covariance_matrix_lower)
        # print(covariance_matrix_upper[0,:])
        # print(covariance_matrix_lower[0,:])

        # ### Find closet 5 points to the uncertain points
        # test_point = 2

        # ### Currently all the points are coming from Xfg
        # dist_array = xvague_sample_current.reshape(-1,2,order='F') - np.vstack((xvague_sample_current.reshape(-1,2,order='F'),Xfz,Xfg))
        # test_point_index = np.argsort(np.sqrt(np.square(dist_array[:,0])+np.square(dist_array[:,1])))[:test_point+1]

        # test_X = np.vstack((xvague_sample_current.reshape(-1,2,order='F'),Xfz,Xfg))[test_point_index[1:],:]
        # test_Y = Y[test_point_index,:]

        # covariance_matrix_upper = compute_K_local(init, x_new.reshape(-1,2,order='F'), test_X) 
        # covariance_matrix_lower = compute_K_local(init, xvague_sample_current.reshape(-1,2,order='F'), test_X)

        # print(test_X)
        # print(test_Y)
        # print(covariance_matrix_upper)
        # print(covariance_matrix_lower)
        # print('---------------------------------------')
        # print(np.linalg.inv(covariance_matrix_upper))
        # print(test_Y.T@np.linalg.inv(covariance_matrix_upper))
        # print(test_Y.T@np.linalg.inv(covariance_matrix_upper)@test_Y)
        # print(-0.5*test_Y.T@np.linalg.inv(covariance_matrix_upper)@test_Y)
        # print('---------------------------------------')
        # print(np.linalg.inv(covariance_matrix_lower))
        # print(test_Y.T@np.linalg.inv(covariance_matrix_lower))
        # print(test_Y.T@np.linalg.inv(covariance_matrix_lower)@test_Y)
        # print(-0.5*test_Y.T@np.linalg.inv(covariance_matrix_lower)@test_Y)

        # likelihood_upper = likelihood(test_Y, covariance_matrix_upper)
        # likelihood_lower = likelihood(test_Y, covariance_matrix_lower)

        print('New:',np.squeeze(x_new),'; Prior:',prior_function_upper,'LH:',likelihood_upper)
        print('Current:',np.squeeze(xvague_sample_current),'; Prior:',prior_function_lower,'LH:',likelihood_lower)
        print('Det ratio:',np.sqrt(np.linalg.det(covariance_matrix_lower)/np.linalg.det(covariance_matrix_upper)))


        accept_ratio = np.sqrt(np.linalg.det(covariance_matrix_lower)/np.linalg.det(covariance_matrix_upper))* \
                        np.exp(prior_function_upper+likelihood_upper-prior_function_lower-likelihood_lower)
        check_sample = np.squeeze(np.random.uniform(0, 1, 1))
        # print('Accept ratio: ',accept_ratio,'; accept sample: ',check_sample)

        if check_sample <= accept_ratio:
            xvague_sample_current = x_new
            xvague_sample_list = np.hstack((xvague_sample_list, xvague_sample_current))
            # print('Accept ratio: ',accept_ratio,'; Xnew: ',x_new,'; Accept')
            print('Accept ratio: ',accept_ratio,'; accept sample: ',check_sample)
            num_samples = num_samples+1
        else:
            # print('Accept ratio: ',accept_ratio,'; Xnew: ',x_new,'; reject')
            pass
    return xvague_sample_list

# trace = run_mcmc(Xfz, Xfg, Y_data, prior_mean, prior_cov, init_params)

def compute_K_local(init, z_prior, Xfz):
    Xuz = z_prior
    params = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    # print(params)
    zz_uu = heat_equation_kuu_noise(Xuz, Xuz, params)  ### K11
    gg_ff = heat_equation_kff(Xfz, Xfz, params) ### K33
    zg_uf = heat_equation_kuf(Xuz, Xfz, params) ### K13
    gz_fu = heat_equation_kfu(Xfz, Xuz, params) ### K31

### Predition
def compute_covariance(init,X1,X2):
    params = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}
    X11 =
    X12 =
    X21 = X12
    X22 = 



def PDE_prediction(init, Xuz, Xfz, Xfg, Y):

    ### Make prediction for z(x)
    ### Glue up the training data (Xuz,Xfz,Xfg)
    X =np.vstack((Xuz,Xfz,Xfg))
    print(X.shape)
    covariance_matrix_upper = compute_K(init, x_new.reshape(-1,2,order='F'), Xfz, Xfg) 


    ### Prior mean is now reshape into a 1D vector t is attached to the end of x.
    num_uncertain_points = Xvague_prior_mean.shape[0]
    Xvague_prior_mean = np.vstack((Xvague_prior_mean[:, 0:1], Xvague_prior_mean[:, 1:2]))
    
    ### Input: num_vague,xvauge_prior_mean, xvague_prior_var
    num_vague = Xvague_prior_mean.shape[0]
    xvague_sample_current = Xvague_prior_mean
    num_samples = 1

    xvague_sample_list = np.empty((num_vague, 0))  ### List of samples
    xvague_sample_list = np.hstack((xvague_sample_list, xvague_sample_current))