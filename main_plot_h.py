# %%
import jax
import datetime
import pickle
import os
from include.plot_pred import plot_and_save_prediction_results, plot_and_save_prediction_results_combine

os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_enable_x64", True)
current_time = datetime.datetime.now().strftime("%m%d")

text = "Pred_Predction_f16_chains1_k0.8_assumption0.02_noisestd0.04_0.0016_k0.8_1100_1021.pkl"
load_path = f"results/datas/trained_params/1021"


# %%
if __name__ == '__main__':
    def load_variables(text, load_path):
        print(f"Loading data from {load_path}")
        filename = f"{text}"

        file_path = os.path.join(load_path, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No data found at {file_path}")
        with open(file_path, 'rb') as f:
            load_variables = pickle.load(f)
        print(f"Variables loaded from {file_path}")
        return load_variables


    variables = load_variables(text, load_path)
    u_values_gt = variables['u_values_gt']
    gp_mean_prior = variables['gp_mean_prior']
    abs_diff_prior = variables['abs_diff_prior']
    gp_mean_posterior = variables['gp_mean_posterior']
    abs_diff_gt_gp = variables['abs_diff_gt_gp']
    var_prior = variables['var_prior']
    var_posterior = variables['var_posterior']
    abs_var_diff = variables['abs_var_diff']
    added_text = variables.get('add_text', text)

# %%
# # %%
    print("start plotting")
    plot_and_save_prediction_results(u_values_gt,
                                     gp_mean_prior,
                                     abs_diff_prior,
                                    gp_mean_posterior,
                                    abs_diff_gt_gp,
                                    var_prior,
                                    var_posterior,
                                    abs_var_diff, added_text)
    plot_and_save_prediction_results_combine(u_values_gt,
                                             gp_mean_prior,
                                             abs_diff_prior,
                                             gp_mean_posterior,
                                             abs_diff_gt_gp,
                                             var_prior,
                                             var_posterior,
                                             abs_var_diff, added_text)


