import GPy
import numpy as np

class GP():
    def __init__(self,kernel_type,noise,gp_lengthscale,gp_variance,message,restart_num):
        self.kernel_type = kernel_type
        self.noise = noise
        self.gp_lengthscale = gp_lengthscale
        self.gp_variance = gp_variance
        self.message = message
        self.restart_num = restart_num
        if self.kernel_type == 'rbf':
            self.kernel = GPy.kern.RBF(input_dim=1, variance=self.gp_variance, lengthscale=self.gp_lengthscale)
    
    def train(self,x,y):
        self.GP = GPy.models.GPRegression(x,y,self.kernel)
        self.GP.Gaussian_noise.fix(self.noise)
        self.GP.optimize(messages=self.message, max_f_eval=1, max_iters=1e7)
        self.GP.optimize_restarts(num_restarts=self.restart_num,verbose=self.message)

    def predict(self,x):
        return self.GP.predict(x)

    def get_parameter(self):
        return self.kernel[0],self.kernel[1]

