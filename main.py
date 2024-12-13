# %%
import jax
import datetime
import optax
from include.heat2d import plot_u_pred
from include.mcmc_posterior import *
from include.init import ModelInitializer_2d
from include.plot_dist import plot_dist, plot_with_noise
from include.train import train_heat_equation_model_2d
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
from scipy.stats import gaussian_kde
import pickle
import os
from test_f_infer_function import plot_f_inference
os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_enable_x64", True)
from include.config import key_num

noise_std = 0.04
prior_std = 0.04
prior_var = prior_std**2# prior variance
max_samples = 2000#2000
assumption_sigma = 0.001#0.001 # step size
k = 0.8#0.5
num_chains = 1
learning_rate = 0.08#0.08
epochs = 1000 #1000


learning_rate_pred = 0.001
epoch_pred= 100
bw=2
num_prior_samples = 200
test_num = 2**4
number_u = 2**2 # xt
number_u_only_x = 4
number_f = 2**4
number_init = 2**3
number_bound = 2**3
number_f_real = (number_f)**2


param_text = "para"
optimizer_in_use = optax.adam
learning = f'lr{learning_rate}&{epochs}'
added_text = f'keynum{key_num}_number_f_real_{number_f_real}_nu{number_u}&{number_u_only_x}&fnumber{number_f}&&b{number_bound}&init{number_init}{learning}&{noise_std}'
weight_decay = 1e-5
mcmc_text = f"noise{noise_std}_prior{prior_std}_maxsamples{max_samples}_assumption{assumption_sigma}_k{k}"
pred_mesh = 200

if __name__ == '__main__':
    print("key_num:", key_num, "\n")
    print("noise_std:", noise_std, "\n")
    print("prior_var:", prior_var, "\n")
    print("number_u:", number_u, "\n")
    print("number_f:", number_f, "\n")
    print("number_init:", number_init, "\n")
    print("number_bound:", number_bound, "\n")
    print("max_samples:", max_samples, "\n")
    print("assumption_sigma:", assumption_sigma, "\n")
    print("k:", k, "\n")
    print("param_text:", param_text, "\n")
    print("optimizer_in_use:", optimizer_in_use, "\n")
    print("epochs:", epochs, "\n")
    print("added_text:", added_text, "\n")
    print("learning_rate1:", learning_rate, "\n")

    print("weight_decay:", weight_decay, "\n")


    model_initializer = ModelInitializer_2d(number_u=number_u, number_f=number_f,
                                            number_init=number_init, number_bound=number_bound, noise_std=noise_std, number_u_only_x=number_u_only_x)
    Xu_certain = model_initializer.Xu_certain
    Xu_noise = model_initializer.Xu_noise
    yu_certain = model_initializer.yu_certain
    print("Xu_certain:", Xu_certain)
    print("Xu_noise:", Xu_noise)
    print("yu_certain:", yu_certain)

    Xu_fixed = model_initializer.Xu_fixed
    Yu_fixed = model_initializer.Yu_fixed
    print("Xu_fixed:", Xu_fixed)
    print("Yu_fixed:", Yu_fixed)

    def plot_points_u(Xu_certain, Xu_noise, Xu_fixed):
        plt.scatter(Xu_certain[:, 0], Xu_certain[:, 1], c="red")
        plt.scatter(Xu_noise[:, 0], Xu_noise[:, 1], c="blue")
        plt.scatter(Xu_fixed[:, 0], Xu_fixed[:, 1], c="green")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.title("Xu")
        current_time = datetime.datetime.now().strftime("%M%S")
        plt.savefig(f"Xu_{current_time}.png")


    Xu_with_noise = model_initializer.Xu_with_noise
    Xu_without_noise = model_initializer.Xu_without_noise
    Yu = model_initializer.Yu
    print("Xu_all_noise:", Xu_with_noise)
    print("Yu:", Yu)

    Xf = model_initializer.Xf
    yf = model_initializer.yf
    print("Xf:", Xf)
    print("yf:", yf)

    def plot_points_f(Xf):
        fig1, ax1 = plt.subplots(1, 1)
        ax1.scatter(Xf[:, 0], Xf[:, 1], c="blue")
        ax1.set_xlabel("x")
        ax1.set_ylabel("t")
        ax1.set_title("Xf")
        current_time = datetime.datetime.now().strftime("%M%S")
        fig1.savefig(f"Xf_{current_time}.png")


    Y = model_initializer.Y
    X_with_noise = model_initializer.X_with_noise

    X_without_noise = model_initializer.X_without_noise
    print("Y:", Y)
    print("X:", X_with_noise)

    number_Y = Y.shape[0]

    init = model_initializer.heat_params_init
    print("Xu_certain:", Xu_certain)
    print("Xu_noise:", Xu_noise)




    print("train init params:", init)






    init = (((jnp.array([50], dtype=jnp.float64),
              jnp.array([0.12, 0.5], dtype=jnp.float64))),)

    print("init params:", init)
    param_iter, optimizer_text, lr_text, epoch_text = train_heat_equation_model_2d(init,
                                                                                   Xu_noise,
                                                                                   Xu_fixed,
                                                                                   Xf,
                                                                                   number_Y,
                                                                                   Y, epochs,
                                                                                   learning_rate,
                                                                                   optimizer_in_use,
                                                                                   mcmc_text
                                                                                   )



    print("param_iter:", param_iter)
    plot_f_inference(pred_mesh, param_iter, Xu_fixed, Yu_fixed, Xf, yf, added_text)

    print('start inference')
    def generate_prior_samples(rng_key, num_samples, prior_mean, prior_cov):
        prior_samples = random.multivariate_normal(rng_key, mean=prior_mean.ravel(), cov=prior_cov,
                                                   shape=(num_samples,))
        prior_samples  = jnp.maximum(jnp.minimum(1, prior_samples ), 0)
        return prior_samples



    def find_kde_peak(data):
        kde = gaussian_kde(data)
        x_vals = jnp.linspace(jnp.min(data), jnp.max(data), 1000)
        kde_vals = kde(x_vals)
        peak_idx = jnp.argmax(kde_vals)
        peak = x_vals[peak_idx]
        return peak


    def find_closest_to_gt(gt, values, labels):
        distances = jnp.abs(jnp.array(values) - gt)
        closest_idx = jnp.argmin(distances)
        closest_label = labels[closest_idx]
        return closest_label, distances


    def save_variables(added_text, **variables):
        root_folder = "."
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        filename = f"{added_text}.pkl"
        file_path = os.path.join(root_folder, filename)

        with open(file_path, 'wb') as f:
            pickle.dump(variables, f)
        print(f"Variables saved to {file_path}")


    prior_rng_key, prior_key = random.split(random.PRNGKey(49))
    prior_cov_flat = jnp.kron(jnp.eye(2) * prior_var, jnp.eye(Xu_noise.shape[0]))
    prior_samples_list = generate_prior_samples(prior_key, num_prior_samples, Xu_noise, prior_cov_flat)
    prior_samples = prior_samples_list.reshape(-1, *Xu_noise.shape)
    print("prior_samples list shape:", prior_samples_list.shape)
    print("prior_samples shape:", prior_samples.shape)

    print(f"assumption_sigma={assumption_sigma}")
    rng_key_chain = jax.random.PRNGKey(422)
    all_chains_samples = []

    for chain_id in range(num_chains):
        rng_key_chain, chain_key = random.split(rng_key_chain)
        chain_samples = single_component_metropolis_hasting(chain_key, max_samples, assumption_sigma, Xu_noise,
                                                            jnp.eye(2) * prior_var, param_iter, Xu_fixed, Xf, Y, k, number_u_only_x)
        all_chains_samples.append(chain_samples)

    all_chains_samples = jnp.array(all_chains_samples)
    num_samples = Xu_noise.shape[0]
    z_uncertain_means = []

    posterior_samples = jnp.concatenate(all_chains_samples, axis=0)
    posterior_samples_list = posterior_samples.reshape(-1, *Xu_noise.shape)


    print("posterior_samples_list shape:", posterior_samples_list.shape)
    print("posterior_samples_list:", posterior_samples_list)
    print("Xu_certain:", Xu_certain)
    print("Xu_noise:", Xu_noise)
    current_time = datetime.datetime.now().strftime("%M%S")
    added_text =  f"chains{num_chains}_f{number_f_real}_k{k}_assumption{assumption_sigma}_prior{prior_std}_noise{noise_std}_maxsamples{max_samples}_numpriorsamples_{num_prior_samples}_learn{learning}_{current_time}"

    Xu_pred_mean = jnp.mean(posterior_samples_list, axis=0)

    plot_u_pred(Xu_without_noise, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred_mean, prior_var,assumption_sigma,k,max_samples,learning,num_chains,number_f,added_text)


    plot_dist(Xu_without_noise, Xu_certain, Xf, Xu_noise, noise_std, Xu_pred_mean, prior_var,assumption_sigma,k,max_samples,learning,num_chains,number_f,posterior_samples_list,
              prior_samples,number_u,added_text)
    plot_with_noise(number_u, number_u_only_x, posterior_samples_list, prior_samples, Xu_certain, Xu_noise, bw,added_text)


    save_variables(added_text, Xu_without_noise=Xu_without_noise, Xu_certain=Xu_certain, Xf=Xf, Xu_noise=Xu_noise,
                   noise_std=noise_std, Xu_pred=Xu_pred_mean, prior_var=prior_var, assumption_sigma=assumption_sigma,
                   k=k, max_samples=max_samples, learning=learning, num_chains=num_chains, number_f=number_f,
                   posterior_samples_list=posterior_samples_list, prior_samples=prior_samples, Y=Y,
                   param_iter=param_iter, Xu_fixed=Xu_fixed, epochs=epochs,
                   learning_rate=learning_rate,
                   optimizer_in_use=optimizer_in_use,number_u=number_u, number_u_only_x=number_u_only_x,prior_std=prior_std,number_bound=number_bound)







