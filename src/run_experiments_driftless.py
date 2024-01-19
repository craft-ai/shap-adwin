from ast import literal_eval
from src.bench_fcts import *
from src.retrain_fcts import *
from src.data_fcts import *
import os
import sys

module_path = os.path.abspath(os.path.join('../../../'))

if module_path not in sys.path:
    sys.path.append(module_path)

####################dataset funcs####################
n_samples = 6000
n_features = 4
np.random.seed(seed=42)


list_df_generate_fcts= []
for noise_rate in [0, 0.001, 0.01, 0.05, 0.1, 0.5]:
    # NOISY 
    def generate_nodrift_df(noise_rate=noise_rate):

        D_G = Drift_generators(n_samples = n_samples, n_features = n_features)
        D_G.abrupt_concept_drift(drifts=[Drift(is_abrupt=True,
                                    start=n_samples-2,
                                    characteristic=[0.7-0.3*(0 % 2) for i in range(D_G.n_features)])])
        D_G.add_noise(noise_rate=noise_rate)
        D_G.drift_name = f"nodrift_noisy{str(noise_rate)[2:]}"
        
        return(D_G)
    list_df_generate_fcts += [generate_nodrift_df]

####################DETECTOR FCTS####################
kswin_funcs = ['retrain_KSWIN_loss',
               'retrain_KSWIN_shap']
adwin_funcs = ['retrain_adwin',
               'retrain_adwin_shap',]
ph_funcs = ['retrain_PH',
            'retrain_PH_shap']


params_retrain_KSWIN = {f: {"alpha": 0.00001, "w_size": 100, "stat_size": 30, "stop_first_detect": True}
                        for f in kswin_funcs}
params_retrain_ADWIN = {f: {"stop_first_detect": True,
                            "delta_d": 0.001} for f in adwin_funcs}  # old: 1
params_retrain_PH = {f: {"stop_first_detect": True, "delta_d": 0.002,  # 0005
                         "min_instances": 30, "threshold": 50, "alpha": 0.9999} for f in ph_funcs}
params_retrain_PH["retrain_PH_shap"] = {"stop_first_detect": True, "delta_d": 0.0001,
                                        "min_instances": 30, "threshold": 50, "alpha": 0.99999}



n_iter = 3
for i,f in enumerate(list_df_generate_fcts):
    print(i, f)
    eval_DG_model_save(f, kswin_funcs, params_retrain_KSWIN, f"KSWIN", n_iter=n_iter)
    eval_DG_model_save(f, adwin_funcs, params_retrain_ADWIN, f"ADWIN", n_iter=n_iter)
    eval_DG_model_save(f, ph_funcs, params_retrain_PH, f"PH", n_iter=n_iter)