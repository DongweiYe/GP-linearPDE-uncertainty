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
Xvague_prior_var = np.ones(num_vague)*prior_var# May result in error when num_vague!=1

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
xvague_sample_current = 0*np.ones(num_vague).reshape(1,-1)
assumption_variance = 5 ### Assumption variance can not be too small as this will define the searching area
xvague_sample_list = np.empty((0,num_vague))
xvague_sample_list = np.vstack((xvague_sample_list,xvague_sample_current))

timestep = 1000
for t in range(timestep):

      ### Important! The workflow below this is now univaraite!!!
      x_new = np.abs(np.random.normal(np.squeeze(xvague_sample_current),assumption_variance,1))
      
      ### Component to compute multivariate Gaussian function for prior
      prior_function_upper = 1/np.sqrt((2*np.pi)*Xvague_prior_var)*\
                              np.exp(-0.5*(x_new-np.squeeze(Xvague_prior_mean))**2/np.squeeze(Xvague_prior_var))
      prior_function_lower = 1/np.sqrt((2*np.pi)*Xvague_prior_var)*\
                              np.exp(-0.5*(xvague_sample_current-np.squeeze(Xvague_prior_mean))**2/np.squeeze(Xvague_prior_var))

      ### Component to compute multivariate Gaussian function for likelihood, note here it is also noice free
      x_upper = np.append(Xexact,x_new)
      x_lower = np.append(Xexact,xvague_sample_current)
      
      y_vector = np.append(yexact,yvague_gt)

      covariance_upper = preGP.kernel.K(np.expand_dims(x_upper,axis=1))
      covariance_lower = preGP.kernel.K(np.expand_dims(x_lower,axis=1))

      determinant_upper = np.linalg.det(covariance_upper)
      determinant_lower = np.linalg.det(covariance_lower)

      likelihood_upper = 1/np.sqrt((2*np.pi)**(num_exact+num_vague)*determinant_upper)*\
                              np.exp(-0.5*np.expand_dims(y_vector,axis=0)@np.linalg.inv(covariance_upper)@np.expand_dims(y_vector,axis=1))
      likelihood_lower = 1/np.sqrt((2*np.pi)**(num_exact+num_vague)*determinant_lower)*\
                              np.exp(-0.5*np.expand_dims(y_vector,axis=0)@np.linalg.inv(covariance_lower)@np.expand_dims(y_vector,axis=1))
      

      upper_alpha = prior_function_upper*likelihood_upper
      lower_alpha = prior_function_lower*likelihood_lower
      print(likelihood_lower)

      accept_ratio = upper_alpha/lower_alpha 
      
      check_sample = np.squeeze(np.random.uniform(0,1,1))

      if check_sample <= accept_ratio:
            xvague_sample_current = x_new
            # print('Test sample accepted',x_new)
            xvague_sample_list = np.vstack((xvague_sample_list,xvague_sample_current))
            print('Accept ratio: ',accept_ratio,'; Xnew: ',x_new,'; Accept')
      else:
            print('Accept ratio: ',accept_ratio,'; Xnew: ',x_new,'; Reject')


print(np.mean(xvague_sample_list[10:,:]),np.var(xvague_sample_list[10:,:]))
print(Xvague_gt)
print(Xvague_prior_mean,Xvague_prior_var)

### Suppose we havea normal distribution g(x|y), here Gaussian distribution.

# ### Assume a prior on the vague point (random mean and some variance)
# random_bias = np.random.rand(num_vague)
# std = 0.5
# vague_X_distribution = 1/np.sqrt(2*np.pi)*std*np.exp(-0.5*((X-Xvague_gt-random_bias)/std)**2)
# print(vague_X_distribution)
