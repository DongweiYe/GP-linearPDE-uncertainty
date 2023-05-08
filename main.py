import numpy as np
import matplotlib.pyplot as plt
import GPy

import bayespy as bp ### Mainly for visualization
import bayespy.plot as bpplt
from scipy.stats import norm

from include.func import *	
from include.vis import *
from include.GP import *
from include.mcmc import *

### fix seed
np.random.seed(5)

### Data parameters for the experiment
output_noise_variance = False
func = FuncClass('exp-sin')      # Define test function

dim = 1                        # Define the dimension of the problem (not encoded))
num_exact = 10                 # number of exact training data
num_vague = 10                 # number of vague training datas
xscale = 8*np.pi               # Scale of input space (array if dim!=1)
prior_var = 1
num_pred = 3

### Create groundtruth data for visualization
X = np.arange(0,xscale,xscale/100)
y = func.run(X,False)

### Create synthetic data 
Xexact = np.random.rand(num_exact)*xscale ### exact training data
yexact = func.run(Xexact,output_noise_variance)

### Create a bunch of vague training data (groundtruth) 
Xvague_gt = np.random.rand(num_vague)*xscale
yvague_gt = func.run(Xvague_gt,output_noise_variance)

### Create the prior of the vague data, isotropic varianace 
Xvague_prior_mean = Xvague_gt+np.random.rand(num_vague)
Xvague_prior_var = np.ones(num_vague)*prior_var# May result in error when num_vague!=1

### Visualization of the problem
# visual_preprocess(X,y,Xexact,yexact,Xvague_gt,yvague_gt,show=True,save=False)

### Train a GP to find the hyperparameters for the kernel in ELBO
preGP = GP('rbf',output_noise=output_noise_variance,gp_lengthscale=1,gp_variance=10,message=False,restart_num=2)
preGP.train(np.expand_dims(np.append(Xexact,Xvague_prior_mean),axis=1),np.expand_dims(np.append(yexact,yvague_gt),axis=1))
y_pred,y_var = preGP.predict(np.expand_dims(X,axis=1))

estimated_noise = preGP.get_noise()
print('Estimated noise: ', estimated_noise)
visual_prediction(X,y,Xexact,yexact,Xvague_gt,yvague_gt,y_pred,show=True,save=False)

### Posterior distribution of input point distributions with MCMC
### With MCMC, samples of posterior distribution wil be genenrated

### Settings for MCMC
### Let the initial guess to be mean of of the prior to each vague data points
### Initial samples for each datapoints
xvague_sample_current = np.multiply(Xvague_prior_mean,np.ones(num_vague)).reshape(1,-1)
assumption_variance = 0.3            ### Assumption variance for jump distribution can not be too small as this will define the searching area
timestep = 1                   ### Artificial timestep

### Bind data for MH computing
databinding = bind_data(Xvague_prior_mean,Xvague_prior_var,Xexact,yexact,yvague_gt,preGP.kernel)

### Perform MCMC with MH algorithm
Xvague_posterior_samplelist = Metropolis_Hasting(timestep,xvague_sample_current,assumption_variance,databinding,output_noise_variance,estimated_noise)
Xvague_posterior_mean = np.mean(Xvague_posterior_samplelist,axis=0)
Xvague_posterior_variance = np.var(Xvague_posterior_samplelist,axis=0)

print('Posterioir mean and variance (Gaussian): ',Xvague_posterior_mean,Xvague_posterior_variance)
print('Prior mean and variance (Gaussian):      ',Xvague_prior_mean,Xvague_prior_var)
print('Groundtruth:                             ',Xvague_gt)

### Visualization of prior, posterior(samples) and groundtruth
plot_distribution(Xvague_prior_mean,Xvague_prior_var,Xvague_posterior_samplelist,Xvague_gt)

### Derive marginalized the posterior distribution over posterior 
### Two way to implmement from this (the posterior of the input locations)
###   1) Directly use the samples from MCMC (currently using this!!!!)
###   2) Computer Gaussian distribution and sample from it

GPvariance,GPlengthscale = preGP.get_parameter()
# ### Generate saving lists for prediction
y_final_mean_list = np.empty((0,X.shape[0]))
y_final_var_list = np.empty((0,X.shape[0]))

# ### Now retrain the GP with new position
# for i in range(100):
for i in range(Xvague_posterior_samplelist.shape[0]):
      local_xtrain = np.append(Xexact,Xvague_posterior_samplelist[i,:])
      local_ytrain = np.append(yexact,yvague_gt)

      postGP = GP('rbf',output_noise_variance,gp_lengthscale=GPlengthscale,gp_variance=GPvariance,message=False,restart_num=5)
      postGP.train(np.expand_dims(local_xtrain,axis=1),np.expand_dims(local_ytrain,axis=1))
      y_final_mean,y_final_var = postGP.predict(np.expand_dims(X,axis=1))
      
      y_final_mean_list = np.vstack((y_final_mean_list,y_final_mean.T))
      y_final_var_list = np.vstack((y_final_var_list,y_final_var.T))

visual_uncertainty(X,y,Xvague_gt,yvague_gt,y_pred,y_var,y_final_mean_list,y_final_var_list,show=True,save=False)

