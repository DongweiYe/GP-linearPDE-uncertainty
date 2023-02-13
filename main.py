import numpy as np
import matplotlib.pyplot as plt

### fix seed
np.random.seed(10)

### Create synthetic data for example
### Create an amplified sin wave, groudtruth
xscale = 8*np.pi
X = np.arange(0,xscale,xscale/100)
y = np.sin(X)*X/2

### Create a bunch of exact training data
Xexact = np.random.rand(10)*xscale
yexact = np.sin(Xexact)*Xexact/2

### Create a bunch of vague training data 
Xvague = np.random.rand(10)*xscale
yvague = np.sin(Xvague)*Xvague/2

### Rescale t
plt.plot(X,y,'k-',label='groundtruth')
plt.plot(Xexact,yexact,'r*',label='exact data')
plt.plot(Xvague,yvague,'b*',label='vague data')
plt.legend()
plt.show()