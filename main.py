import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import torch
import GPy


### fix seed
np.random.seed(10)

### Parameters
num_exact = 14
num_vague = 1

### Create synthetic data for example
### Create an amplified sin wave, groudtruth
xscale = 8*np.pi
X = np.arange(0,xscale,xscale/100)
y = np.sin(X)*X/2

### Create a bunch of exact training data
Xexact = np.random.rand(num_exact)*xscale
yexact = np.sin(Xexact)*Xexact/2

### Create a bunch of vague training data (groundtruth) 
Xvague_gt = np.random.rand(num_vague)*xscale
yvague_gt = np.sin(Xvague_gt)*Xvague_gt/2

### Assume a prior on the vague point (random mean and some variance)
random_bias = np.random.rand(num_vague)
std = 0.5
vague_X_distribution = 1/np.sqrt(2*np.pi)*std*np.exp(-0.5*((X-Xvague_gt-random_bias)/std)**2)
# print(vague_X_distribution)


### Train a GP to find the hyperparameters for the kernel in ELBO
kernel = GPy.kern.RBF(input_dim=1, variance=10., lengthscale=1.)
GP = GPy.models.GPRegression(np.expand_dims(Xexact,axis=1), np.expand_dims(yexact,axis=1), kernel)
GP.Gaussian_noise.fix(1e-4)
GP.optimize(messages=False, max_f_eval=1, max_iters=1e7)
GP.optimize_restarts(num_restarts=3)
y_pred, y_pred_var = GP.predict(np.expand_dims(X,axis=1))




### Visualization
plt.plot(X,y,'k-',linewidth=3,label='groundtruth')
plt.plot(Xexact,yexact,'r*',markersize=10,label='exact data')
plt.plot(Xvague_gt,yvague_gt,'b*',markersize=10,label='vague data ground truth')
plt.plot(X,10*vague_X_distribution+yvague_gt,'g-',label='vague distribution')
plt.plot(X,y_pred,linewidth=3,color='tab:purple',label='GP prediction (exact)')

plt.legend()
plt.show()