import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import torch
import GPy
from include.func import *	
from include.vis import *
from include.GP import *
### fix seed
np.random.seed(10)

### Data parameters for the experiment
func = FuncClass('ampsin')         # Define test function
dim = 1                         # Define the dimension of the problem (not encoded))
num_exact = 14                  # number of exact training data
num_vague = 5                   # number of vague training datas
xscale = 8*np.pi                # Scale of input space (array if dim!=1)

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


### Visualization of the problem
visual_preprocess(X,y,Xexact,yexact,Xvague_gt,yvague_gt,show=True,save=False)

### Train a GP to find the hyperparameters for the kernel in ELBO
preGP = GP('rbf',noise=1e-4,message=False,restart_num=2)
preGP.train(np.expand_dims(Xexact,axis=1),np.expand_dims(yexact,axis=1))
y_pred,y_var = preGP.predict(np.expand_dims(X,axis=1))

visual_prediction(X,y,Xexact,yexact,Xvague_gt,yvague_gt,y_pred,show=True,save=False)

# ### Assume a prior on the vague point (random mean and some variance)
# random_bias = np.random.rand(num_vague)
# std = 0.5
# vague_X_distribution = 1/np.sqrt(2*np.pi)*std*np.exp(-0.5*((X-Xvague_gt-random_bias)/std)**2)
# print(vague_X_distribution)
