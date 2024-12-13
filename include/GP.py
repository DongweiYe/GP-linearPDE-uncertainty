import GPy


### Currently no matter any cases, all training are based on noise
class GP():
    def __init__(self, kernel_type, output_noise, gp_lengthscale, gp_variance, message, restart_num):
        self.kernel_type = kernel_type
        if output_noise == False:
            self.noise_contrain = 1e-4
        else:
            self.noise_contrain = False
        self.gp_lengthscale = gp_lengthscale
        self.gp_variance = gp_variance
        self.message = message
        self.restart_num = restart_num
        if self.kernel_type == 'rbf':
            self.kernel = GPy.kern.RBF(input_dim=1, variance=self.gp_variance, lengthscale=self.gp_lengthscale)

    def train(self, x, y, noise_constraint):
        self.GP = GPy.models.GPRegression(x, y, self.kernel)
        if self.noise_contrain != False:  # Noise free case, add constract of noise
            self.GP.Gaussian_noise.fix(self.noise_contrain)
        if noise_constraint != False:
            self.GP.Gaussian_noise.fix(noise_constraint)
        self.GP.optimize(messages=self.message, max_f_eval=1, max_iters=1e7)
        self.GP.optimize_restarts(num_restarts=self.restart_num, verbose=self.message)

    def predict(self, x):
        return self.GP.predict(x)

    def get_noise(self):
        return self.GP['Gaussian_noise.variance'][0]

    def get_parameter(self):
        return self.kernel[0], self.kernel[1]
