import jax

key = jax.random.PRNGKey(42)

key_x_u, key_x_u_init, key_t_u, key_t_u_low, key_t_u_high, key_x_f, key_t_f, key_x_pred, \
                                                                    key_t_pred, key_x_noise, key_t_noise, \
                                                                    key = jax.random.split(key, 12)


