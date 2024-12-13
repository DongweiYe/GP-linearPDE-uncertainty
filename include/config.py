import jax

# %%
key_num = 100
# key_num = 50
key = jax.random.PRNGKey(key_num)

key_x_rd, key_t_rd, key_x_u, key_x_u_init, key_t_u, key_t_u_low, key_t_u_high, key_x_f, key_t_f, key_x_pred, \
                                                                    key_t_pred, key_x_noise, key_t_noise, \
                                                                    key = jax.random.split(key, 14)

