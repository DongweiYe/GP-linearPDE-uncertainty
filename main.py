import numpy as np
import matplotlib.pyplot as plt
import GPy

# import bayespy as bp ### Mainly for visualization
# import bayespy.plot as bpplt
from scipy.stats import qmc

from include.func import *	
from include.vis import *
from include.GP import *
from include.mcmc import *

### fix seed
np.random.seed(5)

### Data parameters for the experiment
output_noise_variance = 0.2
func = FuncClass('log-sin')      # Define test function

dim = 1                        # Define the dimension of the problem (not encoded))
num_exact = 30                 # number of exact training data
num_vague = 30                 # number of vague training datas
xscale = 8*np.pi               # Scale of input space (array if dim!=1)
prior_var = 1
prior_bias_mean = 0.5

### Create groundtruth data for visualization
X = np.arange(0,xscale,xscale/100)
y = func.run(X,False)

### Create synthetic data (sobol); 2^M
qMCsampler = qmc.Sobol(d=dim,seed=1)
qMCsample = qMCsampler.random_base2(m=6)*xscale

### fetch fixed data
Xexact =  qMCsample[:num_exact,0] ### exact training data
yexact = func.run(Xexact,output_noise_variance)

### Create a bunch of vague training data (groundtruth) 
Xvague_gt = qMCsample[num_exact:(num_exact+num_vague),0]
yvague_gt = func.run(Xvague_gt,output_noise_variance)

### Create the prior of the vague data, isotropic varianace 
Xvague_prior_mean = Xvague_gt+(np.random.rand(num_vague)*2-1)*prior_bias_mean
Xvague_prior_var = np.ones(num_vague)*prior_var# May result in error when num_vague!=1

### Visualization of the problem
# visual_data(X,y,Xexact,yexact,Xvague_gt,Xvague_prior_mean,yvague_gt,show=False,save=True)

### Train a GP to find the hyperparameters for the kernel in ELBO
preGP = GP('rbf',output_noise=output_noise_variance,gp_lengthscale=1,gp_variance=10,message=False,restart_num=2)
preGP.train(np.expand_dims(np.append(Xexact,Xvague_prior_mean),axis=1),np.expand_dims(np.append(yexact,yvague_gt),axis=1),1e-2)
y_pred,y_var = preGP.predict(np.expand_dims(X,axis=1))

estimated_noise = preGP.get_noise()
print('Estimated noise: ', estimated_noise)

# visual_GP(X,y,Xexact,yexact,Xvague_gt,Xvague_prior_mean,yvague_gt,y_pred,y_var,show=True,save=False)
# figure1_standardGP(X,y,Xexact,yexact,Xvague_gt,Xvague_prior_mean,yvague_gt,y_pred,y_var,show=False,save=True)



### Posterior distribution of input point distributions with MCMC
### With MCMC, samples of posterior distribution wil be genenrated

### Settings for MCMC
### Let the initial guess to be mean of of the prior to each vague data points
### Initial samples for each datapoints
xvague_sample_current = np.multiply(Xvague_prior_mean,np.ones(num_vague)).reshape(1,-1)
assumption_variance = 0.01           ### Assumption variance for jump distribution can not be too small as this will define the searching area
timestep = 5000                   ### Artificial timestep

### Bind data for MH computing
databinding = bind_data(Xvague_prior_mean,Xvague_prior_var,Xexact,yexact,yvague_gt,preGP.kernel)

### Perform MCMC with MH algorithm
Xvague_posterior_samplelist = Metropolis_Hasting(timestep,xvague_sample_current,assumption_variance,databinding,output_noise_variance,estimated_noise)
Xvague_posterior_mean = np.mean(Xvague_posterior_samplelist,axis=0)
Xvague_posterior_variance = np.var(Xvague_posterior_samplelist,axis=0)

print('Posterioir mean and variance (Gaussian): ',Xvague_posterior_mean)
print('Prior mean and variance (Gaussian):      ',Xvague_prior_mean)
print('Groundtruth:                             ',Xvague_gt)
print(Xvague_posterior_variance)


### Visualization of prior, posterior(samples) and groundtruth
# plot_distribution(Xvague_prior_mean,Xvague_prior_var,Xvague_posterior_samplelist,Xvague_gt)
# vis_uncertain_input(X,y,Xvague_gt,Xvague_prior_mean,Xvague_posterior_mean,yvague_gt,show=True,save=False)
# figure1_posterior(Xvague_prior_mean,Xvague_prior_var,Xvague_posterior_samplelist,Xvague_gt)


# ### Derive marginalized the predictive distribution over uncertain input posterior 
# ### Derive marginalized the predictive distribution over uncertain input prior (comparison) 


# ### Fetch old hyperparameters
# GPvariance,GPlengthscale = preGP.get_parameter()

# ### Generate saving lists for prediction
# y_final_mean_list = np.empty((0,X.shape[0]))
# y_final_var_list = np.empty((0,X.shape[0]))

# ## Now retrain the GP with uncertain input posterior
# # for i in range(100):
# # if Xvague_posterior_samplelist.shape[0] >= 1500:
# #       Xvague_posterior_samplelist = Xvague_posterior_samplelist[:1000,:]
# for i in range(Xvague_posterior_samplelist.shape[0]):
#       local_xtrain = np.append(Xexact,Xvague_posterior_samplelist[i,:])
#       local_ytrain = np.append(yexact,yvague_gt)

#       postGP = GP('rbf',output_noise_variance,gp_lengthscale=GPlengthscale,gp_variance=GPvariance,message=False,restart_num=5)
#       postGP.train(np.expand_dims(local_xtrain,axis=1),np.expand_dims(local_ytrain,axis=1),False)
#       y_final_mean,y_final_var = postGP.predict(np.expand_dims(X,axis=1))
      
#       y_final_mean_list = np.vstack((y_final_mean_list,y_final_mean.T))
#       y_final_var_list = np.vstack((y_final_var_list,y_final_var.T))


# y_final_mean_list_prior = np.empty((0,X.shape[0]))
# y_final_var_list_prior = np.empty((0,X.shape[0]))

# ### Retrain GP with uncertian input prior
# Xvague_prior_samplelist  = np.random.multivariate_normal(Xvague_prior_mean,np.identity(num_vague)*prior_var,\
#                                                              Xvague_posterior_samplelist.shape[0])

# for i in range(Xvague_posterior_samplelist.shape[0]):
      
#       ### Sample from prior
#       local_xtrain = np.append(Xexact,Xvague_prior_samplelist[i,:])
#       local_ytrain = np.append(yexact,yvague_gt)

#       postGP = GP('rbf',output_noise_variance,gp_lengthscale=GPlengthscale,gp_variance=GPvariance,message=False,restart_num=5)
#       postGP.train(np.expand_dims(local_xtrain,axis=1),np.expand_dims(local_ytrain,axis=1),False)
#       y_final_mean,y_final_var = postGP.predict(np.expand_dims(X,axis=1))
      
#       y_final_mean_list_prior = np.vstack((y_final_mean_list_prior,y_final_mean.T))
#       y_final_var_list_prior = np.vstack((y_final_var_list_prior,y_final_var.T))

# # visual_uncertainty(X,y,Xexact,yexact,Xvague_gt,Xvague_prior_mean,Xvague_posterior_mean,yvague_gt,y_pred,y_var,y_final_mean_list,y_final_var_list,show=True,save=False)
# # vis_prediction(X,y,Xexact,yexact,Xvague_gt,np.mean(Xvague_posterior_samplelist,axis=0),yvague_gt,y_final_mean_list_prior,y_final_var_list_prior,y_final_mean_list,y_final_var_list,show=False,save=True)
# figure2(X,y,Xexact,yexact,Xvague_gt,Xvague_prior_mean,np.mean(Xvague_posterior_samplelist,axis=0),yvague_gt,y_final_mean_list_prior,y_final_var_list_prior,y_final_mean_list,y_final_var_list,show=False,save=True)

# prior_prediction_variance = prediction_variance(y_final_mean_list_prior,y_final_var_list_prior)
# posterior_predition_variance = prediction_variance(y_final_mean_list,y_final_var_list)


print('MSPE position prior: ', (prior_var*num_vague+np.square(np.linalg.norm(Xvague_gt-Xvague_prior_mean)))/num_vague)
print('MSPE position posterior: ',(np.sum(Xvague_posterior_variance)+np.square(np.linalg.norm(Xvague_gt-Xvague_posterior_mean)))/num_vague)

# print('MSPE prediction prior:', (np.sum(prior_prediction_variance)+np.square(np.linalg.norm(y-np.mean(y_final_mean_list_prior,axis=0))))/100)
# print('MSPE prediction posterior:',(np.sum(posterior_predition_variance)+np.square(np.linalg.norm(y-np.mean(y_final_mean_list,axis=0))))/100 )

