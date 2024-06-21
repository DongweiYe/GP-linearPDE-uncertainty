import os
import jax
import datetime

from sklearn.metrics import mean_squared_error
from include.heat2d import f_xt

from include.mcmc_posterior import *

import jax.numpy as jnp
import matplotlib.pyplot as plt

class GaussianProcess:
    def __init__(self, init):
        self.params = {'sigma': init[-1][0], 'lengthscale': init[-1][1]}

    def fit(self, Xfz, Xfg):
        self.Xfz = Xfz
        self.Xfg = Xfg
        zz_ff = heat_equation_kuu(Xfz, Xfz, self.params)
        zg_ff = heat_equation_kuf(Xfz, Xfg, self.params)
        gz_ff = heat_equation_kfu(Xfg, Xfz, self.params)
        gg_ff = heat_equation_kff(Xfg, Xfg, self.params)
        self.K = jnp.block([[zz_ff, zg_ff], [gz_ff, gg_ff]])
        self.K_inv = jnp.linalg.inv(self.K)

    def predict_u(self, Xu_test, y_train):
        zz_ff = heat_equation_kuu(Xu_test, self.Xfz, self.params)
        zg_ff = heat_equation_kuf(Xu_test, self.Xfg, self.params)
        K_s = jnp.block([zz_ff, zg_ff])
        print("k_s shape:", K_s.shape)
        K_ss = heat_equation_kuu(Xu_test, Xu_test, self.params)
        K_inv = self.K_inv
        print("k_inv shape:", K_inv.shape)
        print("y_train shape:", y_train.shape)

        mu_s = K_s.dot(K_inv).dot(y_train)
        cov_s = K_ss - K_s.dot(K_inv).dot(K_s.T)

        return mu_s, cov_s

    def predict_f(self, Xfz_test, y_train):
        zz_ff = heat_equation_kfu(Xfz_test, self.Xfz, self.params)
        zg_ff = heat_equation_kff(Xfz_test, self.Xfg, self.params)
        K_s = jnp.block([zz_ff, zg_ff])
        K_ss = heat_equation_kff(Xfz_test, Xfz_test, self.params)
        K_inv = self.K_inv

        mu_s = K_s.dot(K_inv).dot(y_train)
        cov_s = K_ss - K_s.dot(K_inv).dot(K_s.T)

        return mu_s, cov_s


def check_hyperparamters(init, param_iter, f_xt, Xu_fixed, Yu_fixed, Xf, yf):
    # x_test = Xu_fixed
    # y_test = Yu_fixed
    # gp_init = GaussianProcess(init)
    # Y_train = jnp.concatenate((Yu_fixed, yf))
    # gp_init.fit(Xu_fixed, Xf)
    # y_pred_init, y_cov_init = gp_init.predict_u(x_test, Y_train)
    #
    # gp = GaussianProcess(param_iter)
    # Y_train = jnp.concatenate((Yu_fixed, yf))
    # gp.fit(Xu_fixed, Xf)
    # y_pred, y_cov = gp.predict_u(x_test, Y_train)

    key_test = jax.random.PRNGKey(10)
    Xf_test = jax.random.uniform(key_test, shape=(30, 2), dtype=jnp.float64)
    yf_test = f_xt(Xf_test)
    x_test = Xf_test
    y_test = yf_test
    gp_init = GaussianProcess(init)
    Y_train = jnp.concatenate((Yu_fixed, yf))
    gp_init.fit(Xu_fixed, Xf)
    y_pred_init, y_cov_init = gp_init.predict_f(x_test, Y_train)

    gp = GaussianProcess(param_iter)
    Y_train = jnp.concatenate((Yu_fixed, yf))
    gp.fit(Xu_fixed, Xf)
    y_pred, y_cov = gp.predict_f(x_test, Y_train)

    errors = (y_test - y_pred) ** 2
    errors_init = (y_test - y_pred_init) ** 2
    mse = mean_squared_error(y_test, y_pred)
    mse_init = mean_squared_error(y_test, y_pred_init)
    print(f"Mean Squared Error: {mse}")
    plt.figure(figsize=(10, 6))

    plt.plot(x_test[:, 0], errors, 'bo-', label='Squared Error (Optimized Params)')
    plt.plot(x_test[:, 0], errors_init, 'go-', label='Squared Error (Initial Params)')

    plt.axhline(y=mse, color='b', linestyle='--', label=f'MSE (Optimized Params): {mse:.2f}')
    plt.axhline(y=mse_init, color='g', linestyle='--', label=f'MSE (Initial Params): {mse_init:.2f}')

    plt.xlabel('Test data points')
    plt.ylabel('Squared Error')
    plt.title('Prediction Error (Squared) and Mean Squared Error (MSE)')
    plt.legend()
    current_time = datetime.datetime.now().strftime("%M%S")
    plt.savefig(f"param_iter_f_{current_time}.pdf", format='pdf', bbox_inches='tight')