import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import torch
import GPy
import bayespy as bp
from include.func import *	
from include.vis import *
from include.GP import *
### fix seed
np.random.seed(10)

### Data parameters for the experiment
func = FuncClass('ampsin')         # Define test function
dim = 1                         # Define the dimension of the problem (not encoded))
num_exact = 14                  # number of exact training data
num_vague = 1                   # number of vague training datas
xscale = 8*np.pi                # Scale of input space (array if dim!=1)
prior_var = 0.3

### Create groundtruth data for visualization
X = np.arange(0,xscale,xscale/100)
y = func.run(X)

### Create synthetic data 
Xexact = np.random.rand(num_exact)*xscale ### exact training data
yexact = func.run(Xexact)

### Create a bunch of vague training data (groundtruth) 
Xvague_gt = np.random.rand(num_vague)*xscale
yvague_gt = func.run(Xvague_gt)

### Create the prior of the vague data
Xvague_prior_mean = Xvague_gt+np.random.rand(num_vague)
Xvague_prior_var = np.diag(np.ones(num_vague)*(1/prior_var))# May result in error when num_vague!=1
Xvague_prior = bp.nodes.Gaussian(Xvague_prior_mean,Xvague_prior_var)
print(Xvague_prior)

### Visualization of the problem
visual_preprocess(X,y,Xexact,yexact,Xvague_gt,yvague_gt,show=True,save=False)

### Train a GP to find the hyperparameters for the kernel in ELBO
preGP = GP('rbf',noise=1e-4,message=False,restart_num=2)
preGP.train(np.expand_dims(Xexact,axis=1),np.expand_dims(yexact,axis=1))
y_pred,y_var = preGP.predict(np.expand_dims(X,axis=1))

visual_prediction(X,y,Xexact,yexact,Xvague_gt,yvague_gt,y_pred,show=True,save=False)

### This is a piece of code for MCMC which later should be merge into func.py as a independent function
### Currently only for univariate
### Choose initial point x=0.
xvague_sample_current = np.zeros(num_vague).reshape(1,-1)
assumption_variance = 10
xvague_sample_list = np.empty((0,num_vague))
xvague_sample_list = np.vstack((xvague_sample_list,xvague_sample_current))

timestep = 1
for t in range(timestep):

      ### Important! The workflow below this is now univaraite!!!
      x_new = np.random.normal(np.squeeze(xvague_sample_current),assumption_variance,1)
      
      ### Component to compute multivariate Gaussian function for prior
      prior_function_upper = 1/np.sqrt((2*np.pi)*assumption_variance)*np.exp(-0.5*(x_new-np.squeeze(Xvague_prior_mean))**2/np.squeeze(Xvague_prior_var))
      prior_function_lower = 1/np.sqrt((2*np.pi)*assumption_variance)*np.exp(-0.5*(xvague_sample_current-np.squeeze(Xvague_prior_mean))**2/np.squeeze(Xvague_prior_var))

      ### Component to compute multivariate Gaussian function for likelihood
      
      
#       upper_alpha = 1/np.sqrt()
#       lower_alpha = 
#       accept_ratio = upper_alpha/lower_alpha 
#       check_sample = np.squeeze(np.random.uniform(0,1,1))
#       if check_sample <= accept_ratio:
#             xvague_sample_current = x_new
#       else:
#             pass
#       print(check_sample)


### Suppose we havea normal distribution g(x|y), here Gaussian distribution.

# ### Assume a prior on the vague point (random mean and some variance)
# random_bias = np.random.rand(num_vague)
# std = 0.5
# vague_X_distribution = 1/np.sqrt(2*np.pi)*std*np.exp(-0.5*((X-Xvague_gt-random_bias)/std)**2)
# print(vague_X_distribution)
