import os
import jax
import optax
import pynvml
from include.mcmc_posterior import *
from include.init import ModelInitializer_2d
from include.train import train_heat_equation_model_2d

import jax.numpy as jnp

os.environ["JAX_PLATFORM_NAME"] = "cpu"
#
# Enable 64-bit floating point precision
jax.config.update("jax_enable_x64", True)

optimizer_in_use = optax.adamw

number_u = 20
number_f = 10
number_init = 5
number_bound = 4
sample_num = 10
added_text = f'{number_u}&{number_f}&{sample_num}'
added_text_init = f'init_{number_u}&{number_f}&{sample_num}'
epochs = 1000
learning_rate = 1e-3
weight_decay = 1e-5
DEEP_FLAG = False
prior_std = 1e1
max_samples = 500
assumption_variance = 1e-1

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

    number_u = model_initializer.number_u
    number_f = model_initializer.number_f
    number_init = model_initializer.number_init
    number_bound = model_initializer.number_bound
    number_Y = model_initializer.number_Y

    init = model_initializer.heat_params_init
    print("Xu_noise:", Xu_noise)
    print("Xu_noise shape:", Xu_noise.shape)
    param_iter, optimizer_text, lr_text, epoch_text = train_heat_equation_model_2d(init,
                                                                                   Xu_noise,
                                                                                   Xu_fixed,
                                                                                   Xf,
                                                                                   number_Y,
                                                                                   Y, epochs,
                                                                                   learning_rate,
                                                                                   optimizer_in_use
                                                                                   )



    # posterior_samples_list = Metropolis_Hasting(max_samples, assumption_variance, Xu_noise,
    #                                             jnp.eye(2*number_u)*prior_std**2, param_iter, Xu_fixed, Xf, Y)
    

    print('start inference')
    prior_var = 1e2
    # trace = posterior_inference_mcmc(Xu_noise, jnp.eye(2)*prior_var, param_iter, Xu_fixed, Xf, Y)
    trace = posterior_numpyro(Xu_noise, jnp.eye(2)*prior_var, param_iter, Xu_fixed, Xf, Y)
    posterior_samples = jnp.squeeze(trace.get_values('z_uncertain', combine=True))
    print(posterior_samples.shape)
    print(posterior_samples)

    # print(posterior_samples_list.shape)
    # print(posterior_samples_list)
