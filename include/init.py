import jax.numpy as jnp
from include.config import key_x_u, key_x_f, key_x_u_init, key_t_u_low, key_t_u_high, key_x_noise,\
    key_t_noise
from include.heat2d import get_u_training_data_2d, get_f_training_data_2d, heat_equation_kff, heat_equation_kuf

def initialize_params_2d(sigma_init, lengthscale_init):
    # Initialize RBF kernel hyperparameters
    sigma = jnp.array([sigma_init])
    lengthscale = lengthscale_init

    params = (
        (sigma, lengthscale),
    )

    return params


class ModelInitializer_2d:
    def __init__(self, number_u, number_f, sample_num, number_init, number_bound):
        self.number_u = number_u
        self.number_f = number_f
        self.number_init = number_init
        self.number_bound = number_bound

        self.number_Y = number_u + number_bound * 2 +number_init + number_f

        # Initialize data
        self.Xu, self.yu, self.xu_noise, self.tu_noise, self.Xu_noise, self.yu_noise, self.xu_fixed, self.tu_fixed, \
        self.Xu_fixed, self.Yu_fixed = get_u_training_data_2d(key_x_u, key_x_u_init, key_t_u_low, key_t_u_high,
                                                              key_x_noise, key_t_noise, self.number_u, self.number_init,
                                                              self.number_bound)
        self.xf, self.tf, self.Xf, self.yf = get_f_training_data_2d(key_x_f, self.number_f)

        self.X = jnp.concatenate((self.Xu_noise, self.Xu_fixed, self.Xf))
        self.Y = jnp.concatenate((self.yu, self.Yu_fixed, self.yf))
        self.Y_u = jnp.concatenate((self.yu, self.Yu_fixed))

        # TODO: add mean(yu_noise) to projection
        sigma_init = jnp.std(self.Y)
        sigma_init_yu = jnp.std(self.Y_u)
        sigma_init_yf = jnp.std(self.yf)
        print(f"sigma_init_yu: {sigma_init_yu}", f"sigma_init_yf: {sigma_init_yf}", f"sigma_init: {sigma_init}",
              sep='\t')

        distances_init = jnp.sqrt((self.X[:, None, :] - self.X[None, :, :]) ** 2)
        lengthscale_init = jnp.mean(distances_init, axis=(0, 1))

        kernel_params_only_u = {'sigma': sigma_init, 'lengthscale': lengthscale_init}

        k_ff_inv = jnp.linalg.solve(heat_equation_kff(self.Xf, self.Xf, kernel_params_only_u),
                                    jnp.eye(heat_equation_kff(self.Xf, self.Xf, kernel_params_only_u).shape[0]))
        yf_u = heat_equation_kuf(self.Xf, self.Xf, kernel_params_only_u) @ k_ff_inv @ self.yf
        new_Y = jnp.concatenate((self.yu, yf_u))
        new_sigma_init = jnp.std(new_Y)
        new_sigma_init_yf = jnp.std(yf_u)
        print(f"new_sigma_init_yu: {sigma_init_yu}", f"new_sigma_init_yf: {new_sigma_init_yf}",
              f"new_sigma_init: {new_sigma_init}", sep='\t')

        self.heat_params_init = initialize_params_2d(new_sigma_init, lengthscale_init)

    def initialize(self):
        return self.heat_params_init


