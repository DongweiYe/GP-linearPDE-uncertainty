import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



### fix seed
np.random.seed(10)

### Parameters
num_exact = 20
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
print(vague_X_distribution)


### Train a GP to find the hyperparameters for the kernel in ELBO
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(Xexact, yexact, likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

### Visualization
plt.plot(X,y,'k-',label='groundtruth')
plt.plot(Xexact,yexact,'r*',label='exact data')
plt.plot(Xvague_gt,yvague_gt,'b*',label='vague data ground truth')

plt.plot(X,10*vague_X_distribution+yvague_gt,'g-',label='vague distribution')

plt.legend()
plt.show()