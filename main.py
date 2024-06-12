import os
import jax
import optax
import pynvml
from include.mcmc_posterior import *
from include.init import ModelInitializer_2d
from include.train import train_heat_equation_model_2d

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### The Likelihood is now modified to correct correlation.
### Additional noise is added to diagnoal term for better condition number (for inversion)
### MCMC switch back to Metroplis
### There are issues in computing K -> dimension issue
### Covariance has negative values, not resolved 
### Points are not really correlated.




### fix seed
np.random.seed(0)

os.environ["JAX_PLATFORM_NAME"] = "cpu"
#
# Enable 64-bit floating point precision
jax.config.update("jax_enable_x64", True)

optimizer_in_use = optax.adamw

number_u = 1       ### number of uncertain inputs -> Xuz -> 5
number_f = 100       ### number of fixed input of external source g(x) -> Xfg -> 10
number_init = 50     ### number of initial condition points -> Xfz -> 5
number_bound = 50    ### number of boundary condition points -> Xfz -> 4
sample_num = 10     ### Not used
added_text = f'{number_u}&{number_f}&{sample_num}'
added_text_init = f'init_{number_u}&{number_f}&{sample_num}'
epochs = 1000 ### goes 2000
learning_rate = 1e-3
weight_decay = 1e-5
DEEP_FLAG = False
prior_std = 1e1  ### Prior of uncertainty points


def check_gpu_memory_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory used: {info.used / (1024 ** 2)} MB")
    print(f"GPU memory used: {info.used / (1024 ** 3)} GB")
    pynvml.nvmlShutdown()


if __name__ == '__main__':
    # check_gpu_memory_usage()

    model_initializer = ModelInitializer_2d(number_u=number_u, number_f=number_f, sample_num=sample_num,
                                            number_init=number_init, number_bound=number_bound)
    Xu = model_initializer.Xu
    yu = model_initializer.yu
    xu_noise = model_initializer.xu_noise
    tu_noise = model_initializer.tu_noise
    Xu_noise = model_initializer.Xu_noise
    yu_noise = model_initializer.yu_noise
    xu_fixed = model_initializer.xu_fixed
    tu_fixed = model_initializer.tu_fixed
    Xu_fixed = model_initializer.Xu_fixed
    Yu_fixed = model_initializer.Yu_fixed
    Y = model_initializer.Y
    xf = model_initializer.xf
    tf = model_initializer.tf
    Xf = model_initializer.Xf
    yf = model_initializer.yf

    # print(Xu_fixed)
    # print(Yu_fixed)
    # plt.scatter(Xu_fixed[:,0],Xu_fixed[:,1],c=Yu_fixed,cmap='jet')
    # plt.xlabel('x')
    # plt.ylabel('t')
    # plt.savefig('check_Xu.png',bbox_inches='tight')
    # plt.clf()






    number_u = model_initializer.number_u
    number_f = model_initializer.number_f
    number_init = model_initializer.number_init
    number_bound = model_initializer.number_bound
    number_Y = model_initializer.number_Y

    init = model_initializer.heat_params_init
    print("Noise free Xu:\n",Xu)
    print("Xu_noise:\n", Xu_noise)
    
    param_iter, optimizer_text, lr_text, epoch_text = train_heat_equation_model_2d(init, Xu_noise,Xu_fixed,Xf,
                                                                                   number_Y, Y, epochs,
                                                                                   learning_rate,
                                                                                   optimizer_in_use)
    
    ### Validate the hyperparameters, in this experiement, Xu_noise is noise-free
    PDE_prediction(param_iter,Xu_noise,Xu_fixed, Xf, Y)




    # print('start inference')
    # prior_var = 0.1**2
    # max_samples = 200
    # assumption_variance = 2e-4
    # posterior_samples = Metropolis_Hasting(max_samples, assumption_variance, Xu_noise,
    #                                             jnp.eye(2*number_u)*prior_std**2, param_iter, Xu_fixed, Xf, Y)
    # # trace = run_mcmc(Xu_fixed, Xf, Y, Xu_noise, jnp.eye(2) * prior_var, param_iter, num_samples=2000, num_warmup=500)
    # # posterior_samples = trace
    # # print('Posterior sample shape:',posterior_samples.shape)
    # num_samples = Xu_noise.shape[0]
    # z_uncertain_means = []

    # posterior_samples_list = np.mean(posterior_samples, axis=1)
    # print(posterior_samples.shape)
    # print(posterior_samples_list)
    # for vague_points in range(number_u):
    #     fig = plt.figure()
    #     plt.axvline(Xu[vague_points, 0], color='tab:red',label='GT')
    #     plt.axvline(Xu_noise[vague_points, 0], color='tab:green',label='noised')
    #     plt.axvline(posterior_samples_list[vague_points], color='tab:blue',label='denoised')
    #     # sns.kdeplot(posterior_samples_list[vague_points], color='tab:blue',label='denoised')
    #     plt.legend()
    #     plt.savefig(f"kdeplot_x_{vague_points}.png",bbox_inches='tight')
    #     plt.close()

    # for vague_points in range(number_u):
    #     fig = plt.figure()
    #     plt.axvline(Xu[vague_points, 1], color='tab:red',label='GT')
    #     plt.axvline(Xu_noise[vague_points, 1], color='tab:green',label='noised')
    #     plt.axvline(posterior_samples_list[vague_points], color='tab:blue',label='denoised')
    #     # sns.kdeplot(posterior_samples_list[vague_points+number_u], color='tab:blue',label='denoised')
    #     plt.legend()
    #     plt.savefig(f"kdeplot_t_{vague_points}.png",bbox_inches='tight')
    #     plt.close()


    # for i in range(num_samples):
    #     param_name = f'z_uncertain{i}'
    #     z_uncertain_mean = np.mean(posterior_samples[param_name], axis=0)
    #     print(f"{i}:_z_uncertain_mean={z_uncertain_mean}")
    #     z_uncertain_means.append(z_uncertain_mean)

    # for i in range(num_samples):
    #     param_name = f'z_uncertain{i}'
    #     if param_name in posterior_samples:
    #         data = posterior_samples[param_name]
    #         print(data.shape)
    #         for j in range(2):
    #             plt.figure(figsize=(8, 5))
    #             if j == 0:
    #                 plt.xlabel('uncertain x of point '+str(i))
    #             else:
    #                 plt.xlabel('uncertain t of point '+str(i))
    #             plt.ylabel('Density')
    #             plt.title(f'Posterior Distribution of {param_name}')
    #             plt.axvline(Xu[i, j], color='tab:red', label='Xu True Value')
    #             plt.axvline(Xu_noise[i, j], color='tab:green', label='Xu Noise Value')
    #             sns.kdeplot(data[:,j], color='tab:blue', label=f'{param_name} Posterior')
    #             # sns.kdeplot(posterior_samples['z_uncertain' + str(i)], color='tab:blue', label='Posterior KDE')

    #             plt.legend()
    #             # plt.savefig(f"hist_posterior_{param_name}.pdf", format='pdf')
    #             plt.savefig(f"kde_posterior_{param_name}_{j}.png",bbox_inches='tight')

    #     else:
    #         print(f'Key {param_name} not found in posterior_samples.')


    