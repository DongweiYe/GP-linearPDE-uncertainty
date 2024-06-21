import datetime
from functools import partial

import jax
import matplotlib.pyplot as plt
import optax

from include.heat2d import heat_equation_nlml_loss_2d

def train_heat_equation_model_2d(heat_params_init, Xuz, Xfz, Xfg, number_Y, Y, num_epochs, learning_rate,
                                 optimizer_in_use):
    fig = plt.figure()
    ax = fig.add_subplot(111, title="Training")
    plt.ylabel("nlml loss")

    scheduler_exp_decay = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=num_epochs,
        decay_rate=0.9
    )
    final_scheduler = scheduler_exp_decay

    def train_step(inner_params, inner_opt_state):
        grad = grad_fn(inner_params)
        updates, inner_opt_state = opt.update(grad, inner_opt_state, params=inner_params)
        new_params = optax.apply_updates(inner_params, updates)
        return new_params, inner_opt_state

    param_iter = heat_params_init
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

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        current_learning_rate = final_scheduler(epoch)
        # print(f"Step {epoch}: Learning rate = {current_learning_rate:.6f}")
        learning_rates.append(current_learning_rate)
        param_iter, opt_state = train_step(param_iter, opt_state)
        loss = loss_fn(param_iter)
        y_loss = loss
        # print(f"Epoch {epoch+1} Loss: {loss}")
        losses.append(y_loss)
        # ax.plot(epoch, y_loss, 'bo-', label='train')

    def print_optimizer_info(fig, learning_rate=learning_rate, optimizer_in_use_name=optimizer_in_use_name,
                             num_epochs=num_epochs):
        lr_text = f"{learning_rate:.6f}"
        optimizer_text = f"{optimizer_in_use_name}"
        epoch_text = f"{num_epochs}"
        fig.text(0.02, 0.98, f"{optimizer_text}\n{lr_text}\n{epoch_text}", ha='left', va='top', fontsize=10)
        print(f"{optimizer_text}\n{lr_text}\n{epoch_text}")


    # plt.ylim(-100, 100)
    ax.plot(losses, 'bo-', label='train')
    plt.xlabel(f"loss = {loss}")
    print_optimizer_info(fig)
    lr_text = f"{learning_rate:.6f}"
    optimizer_text = f"{optimizer_in_use_name}"
    epoch_text = f"{num_epochs}"
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    # plt.savefig(f"train_{optimizer_text}_{lr_text}_{epoch_text}_{current_time}.pdf", format='pdf')
    # print(f"Initial loss: {init_loss}")
    # print(f"Final loss: {loss}")
    return param_iter, optimizer_text, lr_text, epoch_text

