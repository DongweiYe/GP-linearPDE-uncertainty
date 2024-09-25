import datetime
from functools import partial

import jax
import matplotlib.pyplot as plt
import optax
import jax.numpy as jnp
from include.heat2d import heat_equation_nlml_loss_2d, heat_equation_nlml_loss_2d_rd


def normalize_data_pair(x, y):
    x_mean = jnp.mean(x, axis=0)
    x_std = jnp.std(x, axis=0)

    y_mean = jnp.mean(y)
    y_std = jnp.std(y)

    x_normalized = (x - x_mean) / x_std

    y_normalized = (y - y_mean) / y_std

    return x_normalized, y_normalized


def normalize_x(x):
    x_mean = jnp.mean(x, axis=0)
    x_std = jnp.std(x, axis=0)
    x_normalized = (x - x_mean) / x_std
    return x_normalized


def normalize_y(y):
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)
    y_normalized = (y - y_mean) / y_std

    return y_normalized

def train_heat_equation_model_2d(heat_params_init, Xuz, Xfz, Xfg, number_Y, Y, num_epochs, learning_rate,
                                 optimizer_in_use,mcmc_text):
    fig = plt.figure()
    ax = fig.add_subplot(111, title="Training")
    plt.ylabel("nlml loss")

    scheduler_exp_decay = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=num_epochs,
        decay_rate=0.9
    )
    final_scheduler = scheduler_exp_decay

    # scheduler_poly_decay = optax.polynomial_schedule(
    # init_value=learning_rate,
    # end_value=0.001,
    # power=2.0,
    # transition_steps=num_epochs
    # )
    # final_scheduler = scheduler_poly_decay
    #
    # boundaries = [int(0.3 * num_epochs), int(0.6 * num_epochs), int(0.8 * num_epochs)]
    # values = [learning_rate, learning_rate * 0.1, learning_rate * 0.01, learning_rate * 0.001]
    # scheduler_piecewise_constant = optax.piecewise_constant_schedule(
    #     init_value=learning_rate,
    #     boundaries_and_scales=dict(zip(boundaries, values))
    # )
    # final_scheduler = scheduler_piecewise_constant
    #
    # warmup_steps = int(0.1 * num_epochs)
    # scheduler_warmup_exponential_decay = optax.warmup_exponential_decay_schedule(
    #     init_value=0.001,
    #     peak_value=learning_rate,
    #     warmup_steps=warmup_steps,
    #     transition_steps=num_epochs - warmup_steps,
    #     decay_rate=0.9
    # )
    # final_scheduler = scheduler_warmup_exponential_decay

    # scheduler_cosine_decay = optax.cosine_decay_schedule(
    #     init_value=learning_rate,
    #     decay_steps=num_epochs
    # )
    # final_scheduler = scheduler_cosine_decay

    # scheduler_constant = optax.constant_schedule(value=learning_rate)
    # final_scheduler = scheduler_constant

    def train_step(inner_params, inner_opt_state):
        grad = grad_fn(inner_params)
        updates, inner_opt_state = opt.update(grad, inner_opt_state, params=inner_params)
        new_params = optax.apply_updates(inner_params, updates)
        return new_params, inner_opt_state

    param_iter = heat_params_init

    # Xuz_normalized = normalize_x(Xuz)
    # Xfz_normalized = normalize_x(Xfz)
    # Xfg_normalized = normalize_x(Xfg)
    # Y_normalized = normalize_y(Y)
    # loss_fn = partial(heat_equation_nlml_loss_2d, Xuz=Xuz_normalized, Xfz=Xfz_normalized, Xfg=Xfg_normalized,
    #                   number_Y=number_Y, Y=Y_normalized)

    loss_fn = partial(heat_equation_nlml_loss_2d, Xuz=Xuz, Xfz=Xfz, Xfg=Xfg, number_Y=number_Y, Y=Y)
    grad_fn = jax.grad(loss_fn)
    init_loss = loss_fn(param_iter)

    optimizer_in_use = optimizer_in_use
    optimizer_in_use_name = getattr(optimizer_in_use, '__name__')
    opt = optax.chain(
        optax.clip(1.0),
        optax.scale_by_schedule(final_scheduler),
        optimizer_in_use(learning_rate=learning_rate)
    )

    opt_state = opt.init(param_iter)
    learning_rates = []
    losses = []

    for epoch in range(num_epochs):
        current_learning_rate = final_scheduler(epoch)
        # print(f"Step {epoch}: Learning rate = {current_learning_rate:.6f}")
        learning_rates.append(current_learning_rate)
        param_iter, opt_state = train_step(param_iter, opt_state)
        loss = loss_fn(param_iter)
        y_loss = loss
        print(f"Epoch {epoch+1} Loss: {loss}")
        losses.append(y_loss)
        ax.plot(epoch, y_loss, 'bo-', label='train')



    plt.ylim(-5000, 4000)
    ax.plot(losses, 'bo-', label='train')
    plt.xlabel(f"loss = {loss}")
    # print_optimizer_info(fig)
    lr_text = f"{learning_rate:.6f}"
    optimizer_text = f"{optimizer_in_use_name}"
    epoch_text = f"{num_epochs}"
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    plt.savefig(f"train_{lr_text}_{epoch_text}_{mcmc_text}_{current_time}.pdf", format='pdf')
    print(f"Initial loss: {init_loss}")
    print(f"Final loss: {loss}")
    plt.close(fig)
    return param_iter, optimizer_text, lr_text, epoch_text


def train_heat_equation_model_2d_rd(heat_params_init, Xuz, Xfz, Xfg, number_Y, Y, num_epochs, learning_rate,
                                 optimizer_in_use,mcmc_text):
    fig = plt.figure()
    ax = fig.add_subplot(111, title="Training")
    plt.ylabel("nlml loss")

    scheduler_exp_decay = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=num_epochs,
        decay_rate=0.9
    )
    final_scheduler = scheduler_exp_decay

    # scheduler_poly_decay = optax.polynomial_schedule(
    # init_value=learning_rate,
    # end_value=0.001,
    # power=2.0,
    # transition_steps=num_epochs
    # )
    # final_scheduler = scheduler_poly_decay
    #
    # boundaries = [int(0.3 * num_epochs), int(0.6 * num_epochs), int(0.8 * num_epochs)]
    # values = [learning_rate, learning_rate * 0.1, learning_rate * 0.01, learning_rate * 0.001]
    # scheduler_piecewise_constant = optax.piecewise_constant_schedule(
    #     init_value=learning_rate,
    #     boundaries_and_scales=dict(zip(boundaries, values))
    # )
    # final_scheduler = scheduler_piecewise_constant
    #
    # warmup_steps = int(0.1 * num_epochs)
    # scheduler_warmup_exponential_decay = optax.warmup_exponential_decay_schedule(
    #     init_value=0.001,
    #     peak_value=learning_rate,
    #     warmup_steps=warmup_steps,
    #     transition_steps=num_epochs - warmup_steps,
    #     decay_rate=0.9
    # )
    # final_scheduler = scheduler_warmup_exponential_decay

    # scheduler_cosine_decay = optax.cosine_decay_schedule(
    #     init_value=learning_rate,
    #     decay_steps=num_epochs
    # )
    # final_scheduler = scheduler_cosine_decay

    # scheduler_constant = optax.constant_schedule(value=learning_rate)
    # final_scheduler = scheduler_constant

    def train_step(inner_params, inner_opt_state):
        grad = grad_fn(inner_params)
        updates, inner_opt_state = opt.update(grad, inner_opt_state, params=inner_params)
        new_params = optax.apply_updates(inner_params, updates)
        return new_params, inner_opt_state

    param_iter = heat_params_init

    # Xuz_normalized = normalize_x(Xuz)
    # Xfz_normalized = normalize_x(Xfz)
    # Xfg_normalized = normalize_x(Xfg)
    # Y_normalized = normalize_y(Y)
    # loss_fn = partial(heat_equation_nlml_loss_2d, Xuz=Xuz_normalized, Xfz=Xfz_normalized, Xfg=Xfg_normalized,
    #                   number_Y=number_Y, Y=Y_normalized)

    loss_fn = partial(heat_equation_nlml_loss_2d_rd, Xuz=Xuz, Xfz=Xfz, Xfg=Xfg, number_Y=number_Y, Y=Y)
    grad_fn = jax.grad(loss_fn)
    init_loss = loss_fn(param_iter)

    optimizer_in_use = optimizer_in_use
    optimizer_in_use_name = getattr(optimizer_in_use, '__name__')
    opt = optax.chain(
        optax.clip(1.0),
        optax.scale_by_schedule(final_scheduler),
        optimizer_in_use(learning_rate=learning_rate)
    )

    opt_state = opt.init(param_iter)
    learning_rates = []
    losses = []

    for epoch in range(num_epochs):
        current_learning_rate = final_scheduler(epoch)
        # print(f"Step {epoch}: Learning rate = {current_learning_rate:.6f}")
        learning_rates.append(current_learning_rate)
        param_iter, opt_state = train_step(param_iter, opt_state)
        loss = loss_fn(param_iter)
        y_loss = loss
        print(f"Epoch {epoch+1} Loss: {loss}")
        losses.append(y_loss)
        ax.plot(epoch, y_loss, 'bo-', label='train')



    plt.ylim(-5000, 4000)
    ax.plot(losses, 'bo-', label='train')
    plt.xlabel(f"loss = {loss}")
    # print_optimizer_info(fig)
    lr_text = f"{learning_rate:.6f}"
    optimizer_text = f"{optimizer_in_use_name}"
    epoch_text = f"{num_epochs}"
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    # plt.savefig(f"train_{lr_text}_{epoch_text}_{mcmc_text}_{current_time}.pdf", format='pdf')
    print(f"Initial loss: {init_loss}")
    print(f"Final loss: {loss}")
    plt.close(fig)
    return param_iter, optimizer_text, lr_text, epoch_text
