import jax.numpy as jnp
from include.config import key_x_u, key_x_f, key_x_u_init, key_t_u_low, key_t_u_high, key_x_noise,\
    key_t_noise
from include.heat2d import get_f_training_data_2d, \
    get_u_training_data_2d_qmc, compute_kuf, compute_kff


def initialize_params_2d(sigma_init, lengthscale_init):
    sigma = jnp.array([sigma_init])
    lengthscale = lengthscale_init

    params = (
        (sigma, lengthscale),
    )

    return params


class ModelInitializer_2d:
    def __init__(self, number_u, number_f, number_inner_b, number_init, number_bound, noise_std, number_u_only_x):
        self.number_u = number_u
        self.number_f = number_f
        self.number_init = number_init
        self.number_inner_b = number_inner_b
        self.number_bound = number_bound
        self.noise_std = noise_std
        self.number_u_only_x = number_u_only_x
        # self.number_Y = number_u + number_bound * 2 + number_init + number_f

        # Initialize data
        self.Xu_certain, self.yu_certain, self.xu_noise, self.tu_noise, self.Xu_noise, self.xu_fixed, self.tu_fixed, \
        self.Xu_fixed, self.Yu_fixed, self.number_init, self.number_inner_b, self.number_bound = get_u_training_data_2d_qmc(key_x_u, key_x_u_init, key_t_u_low, key_t_u_high,
                                                              key_x_noise, key_t_noise, self.number_u, self.number_init, self.number_inner_b,
                                                              self.number_bound, self.noise_std, self.number_u_only_x)
        self.xf, self.tf, self.Xf, self.yf = get_f_training_data_2d(key_x_f, self.number_f)

        self.X_with_noise = jnp.concatenate((self.Xu_noise, self.Xu_fixed, self.Xf))
        self.X_without_noise = jnp.concatenate((self.Xu_certain, self.Xu_fixed, self.Xf))
        self.Y = jnp.concatenate((self.yu_certain, self.Yu_fixed, self.yf))
        self.number_Y = self.Y.shape[0]
        self.Yu = jnp.concatenate((self.yu_certain, self.Yu_fixed))
        self.Xu_with_noise = jnp.concatenate((self.Xu_noise, self.Xu_fixed))
        self.Xu_without_noise = jnp.concatenate((self.Xu_certain, self.Xu_fixed))

        self.xtest = jnp.concatenate((self.Xu_fixed, self.Xf))
        self.ytest = jnp.concatenate((self.Yu_fixed, self.yf))

        sigma_init = jnp.std(self.Y)
        sigma_init_yu = jnp.std(self.Yu)
        sigma_init_yf = jnp.std(self.yf)
        print(f"sigma_init_yu: {sigma_init_yu}", f"sigma_init_yf: {sigma_init_yf}", f"sigma_init: {sigma_init}",
              sep='\t')

        distances_init = jnp.sqrt((self.Xu_with_noise[:, None, :] - self.Xu_with_noise[None, :, :]) ** 2)
        lengthscale_init = jnp.mean(distances_init, axis=(0, 1))

        #kernel_params_only_u = {'sigma': sigma_init, 'lengthscale': lengthscale_init}
        kernel_params_only_u = initialize_params_2d(sigma_init_yu, lengthscale_init)
        lengthscale_x = kernel_params_only_u[0][1][0].item()
        lengthscale_t = kernel_params_only_u[0][1][1].item()
        k_ff = compute_kff(self.Xf, self.Xf, kernel_params_only_u, lengthscale_x, lengthscale_t)
        k_ff_inv_yf: jnp.ndarray = jnp.linalg.solve(k_ff, self.yf)
        yf_u =compute_kuf(self.Xf, self.Xf, kernel_params_only_u, lengthscale_x, lengthscale_t) @ k_ff_inv_yf
        # k_ff_inv = jnp.linalg.solve(heat_equation_kff(self.Xf, self.Xf, kernel_params_only_u),
        #                             jnp.eye(heat_equation_kff(self.Xf, self.Xf, kernel_params_only_u).shape[0]))
        # yf_u = heat_equation_kuf(self.Xf, self.Xf, kernel_params_only_u) @ k_ff_inv @ self.yf
        new_Y = jnp.concatenate((self.yu_certain, yf_u))
        new_sigma_init = jnp.std(new_Y)
        new_sigma_init_yf = jnp.std(yf_u)
        print(f"new_sigma_init_yu: {sigma_init_yu}", f"new_sigma_init_yf: {new_sigma_init_yf}",
              f"new_sigma_init: {new_sigma_init}", sep='\t')

        self.heat_params_init = initialize_params_2d(new_sigma_init, lengthscale_init)
        #self.heat_params_init = initialize_params_2d(sigma_init_yu, lengthscale_init)

        print("use %%%new_sigma_init%%%% for kernel_params")
    def initialize(self):
        return self.heat_params_init


