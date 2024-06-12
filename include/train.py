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

    def train_step(heat_params, opt_state):
        grad = grad_fn(heat_params)
        updates, opt_state = opt.update(grad, opt_state, params=heat_params)
        new_heat_params = optax.apply_updates(heat_params, updates)
        return new_heat_params, opt_state

    param_iter = heat_params_init
    loss_fn = partial(heat_equation_nlml_loss_2d, Xuz=Xuz, Xfz=Xfz, Xfg=Xfg, number_Y=number_Y, Y=Y)
    grad_fn = jax.grad(loss_fn)
    init_loss = loss_fn(param_iter)
    optimizer_in_use = optimizer_in_use
    optimizer_in_use_name = getattr(optimizer_in_use, '__name__')
    opt = optax.chain(
        optax.clip(1.0),
        optimizer_in_use(learning_rate=learning_rate)
    )

    num_epochs = num_epochs
    opt_state = opt.init(param_iter)

    learning_rates = []
    losses = []

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
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
    plt.savefig(f"train_{optimizer_text}_{lr_text}_{epoch_text}_{current_time}.pdf", format='pdf')
    print(f"Initial loss: {init_loss}")
    print(f"Final loss: {loss}")
    return param_iter, optimizer_text, lr_text, epoch_text

