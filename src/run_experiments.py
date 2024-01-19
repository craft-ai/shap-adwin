from ast import literal_eval
from src.bench_fcts import *
from src.retrain_fcts import *
from src.data_fcts import *
import os
import sys

####################dataset funcs####################
n_samples = 6000
n_features = 4
np.random.seed(seed=42)

list_df_generate_fcts = []
for noise_rate in [0, 0.001, 0.01, 0.05, 0.1, 0.5]:
    # NOISY
    def generate_noisy_smooth_concept_drift(n_samples=n_samples, noise_rate=noise_rate):
        D_G = Drift_generators(n_samples=n_samples, n_features=n_features)
        D_G.smooth_concept_drift(n_drift=1)
        D_G.add_noise(noise_rate=noise_rate)
        return (D_G)

    def generate_noisy_abrupt_concept_drift(n_samples=n_samples, noise_rate=noise_rate):
        D_G = Drift_generators(n_samples=n_samples, n_features=n_features)
        D_G.abrupt_concept_drift(n_drift=1)
        D_G.add_noise(noise_rate=noise_rate)
        return (D_G)

    def generate_noisy_gradual_concept_drift(n_samples=n_samples, noise_rate=noise_rate):
        D_G = Drift_generators(n_samples=n_samples, n_features=n_features)
        D_G.gradual_concept_drift(n_drift=1)
        D_G.add_noise(noise_rate=noise_rate)
        return (D_G)

    def generate_noisy_abrupt_covariate_drift(n_samples=n_samples, noise_rate=noise_rate):
        D_G = Drift_generators(n_samples=n_samples, n_features=n_features)
        d_centers = [(0, 0,
                      0.3, 0.25, 1), (0, 0.3, 0.5, 0, 0.25), (0, 0.5, 1, 0.25, 1)]
        D_G.abrupt_covariate_drift(d_centers=d_centers)
        D_G.add_noise(noise_rate=noise_rate)
        return (D_G)

    def generate_noisy_stagger(noise_rate=noise_rate):
        D_G = Drift_generators()
        D_G.load_df("stagger_short")
        D_G.add_noise(noise_rate=noise_rate)
        return (D_G)
#

    def generate_noisy_sine1(noise_rate=noise_rate):
        D_G = Drift_generators()
        D_G.load_df("sine1_short")
        D_G.add_noise(noise_rate=noise_rate)
        return (D_G)
#

    def generate_noisy_sine2(noise_rate=noise_rate):
        D_G = Drift_generators()
        D_G.load_df("sine2_short")
        D_G.add_noise(noise_rate=noise_rate)
        return (D_G)

        # generate_noisy_smooth_concept_drift
    noisy_fcts = [generate_noisy_smooth_concept_drift, generate_noisy_abrupt_concept_drift, generate_noisy_abrupt_covariate_drift,
                  generate_noisy_sine1, generate_noisy_sine2, generate_noisy_stagger]

    list_df_generate_fcts += noisy_fcts

####################DETECTOR FCTS####################
kswin_funcs = ['retrain_KSWIN_loss',
               'retrain_KSWIN_shap']
adwin_funcs = ['retrain_adwin',
               'retrain_adwin_shap',]
ph_funcs = ['retrain_PH',
            'retrain_PH_shap']


params_retrain_KSWIN = {f:{"alpha":0.00095, "w_size":100, "stat_size":30, "stop_first_detect" :True}
                        for f in kswin_funcs}
params_retrain_KSWIN["retrain_KSWIN_shap"]  = {"alpha":0.00024, "w_size":100, "stat_size":30, "stop_first_detect" :True}

params_retrain_ADWIN = {f: {"stop_first_detect": True,
                            "delta_d": 0.001} for f in adwin_funcs}  # old: 1
params_retrain_ADWIN["retrain_adwin_shap"] = {"stop_first_detect" :True, "delta_d":0.1}

params_retrain_PH = {f:{"stop_first_detect" :True, "delta_d":0.0001,
                       "min_instances":30,"threshold":11, "alpha": 0.9999} for f in ph_funcs}
params_retrain_PH["retrain_PH_shap"] = {"stop_first_detect": True, "delta_d": 0.0001,
                                        "min_instances": 30, "threshold": 50, "alpha": 0.99999}


n_iter = 3
for i,f in enumerate(list_df_generate_fcts):
    print(i, f)
    eval_DG_model_save(f, kswin_funcs, params_retrain_KSWIN, "KSWIN", n_iter=n_iter)
    eval_DG_model_save(f, adwin_funcs, params_retrain_ADWIN, "ADWIN", n_iter=n_iter)
    eval_DG_model_save(f, ph_funcs, params_retrain_PH, "PH", n_iter=n_iter)