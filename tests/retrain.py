# %%
import os, sys

module_path = os.path.abspath(os.path.join('../'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.data_fcts import *
from src.retrain_fcts import *

# %%

import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown]
# # TEST SHAP

# %%


# %%
funcs = ['retrain_adwin',
         'retrain_adwin_loss',
         'retrain_adwin_loss',
         ]

# %%
np.random.randint(1000)

# %%
D_G = Drift_generators(n_samples = 3000, n_features = 3, feature_random_seed =40)
D_G.abrupt_concept_drift(drifts=[Drift(True, start= 1000, characteristic=[0.8,0.8,0.8])])
Retrainer(D_G).plot_retrain()

# %%
from src.shap_fcts import init_shap, compute_shap_val

# %%

Rt = Retrainer(D_G)
explainer = init_shap("shap", Rt, bgd_type="sample", n_samp_bgd=20)



# %%
for index, (x, label) in enumerate(zip(Rt.X_unseen.values, Rt.y_unseen.values)):

    break

# %%
x

# %%
explainer.shap_values(X=np.array(x), y=[label])

# %%
import shap

# %%
Rt = Retrainer(D_G)
Rt.retrain_detector_shap(detector_func=ADWIN,
                                     signal="log_loss",
                                     detector_params={
                                         "delta": 0.01, "clock": 1},
                                     warning_params={
                                         "delta": 0.2, "clock": 1},
                                     win_size=20,
                                     retrain_name="adwin_loss_shap");

print(Rt.detection_indices)

# %%
print(Rt.detection_indices)
Rt.plot_retrain(with_no_retrain=True)

# %%


# %%


# %%
Rt = Retrainer(D_G)
#for bgd in ["worse", "best", "sample", "train", "small"]:
Rt.retrain_adwin_shap(bgd_type="best");

# %%
for bgd in ["worse", "best", "sample", "train", "small"][::-1]:
    Rt = Retrainer(D_G)
    Rt.retrain_adwin_shap(bgd_type=bgd);
    print(Rt.detection_indices)


# %%
print(Rt.detection_indices)
Rt.plot_retrain(with_no_retrain=True)

# %%
#funcs = ["retrain_periodic", "retrain_adwin_loss", "retrain_adwin"]
fig, ax = plt.subplots(nrows=len(funcs), ncols=1, figsize = [20,6*len(funcs)])
for i_f, fct in enumerate(funcs):
    Rt = Retrainer(D_G)
    if("period" in fct):
        getattr(Retrainer, fct)(Rt, width= 250, period=400)
    else:
        getattr(Retrainer, fct)(Rt)
    print(fct, Rt.detection_indices)
    Rt.plot_retrain(with_no_retrain=True, ax=ax[i_f])

# %% [markdown]
# # Other

# %%
from river.drift import ADWIN 

# %%
funcs = [func for func in dir(Retrainer) if callable(getattr(Retrainer, func))
         and not func.startswith("__")
         and "retrain" in func and not "plot" in func and not "perf" in func]

# %%
[x for x in funcs if "shap" not in x]

# %%


# %%
funcs = ["retrain_periodic",
        'retrain_adwin',
        'retrain_adwin_loss',
        "retrain_PH",
        "retrain_PH_loss",
        "retrain_KSWIN",
        "retrain_KSWIN_loss",]

# %%
np.random.randint(1000)

# %%
D_G = Drift_generators(n_samples = 3000, n_features = 3, feature_random_seed =40)
D_G.abrupt_concept_drift(drifts=[Drift(True, start= 1000, characteristic=[0.8,0.8,0.8])])
Retrainer(D_G).plot_retrain()

# %%
#funcs = ["retrain_periodic", "retrain_adwin_loss", "retrain_adwin"]
fig, ax = plt.subplots(nrows=len(funcs), ncols=1, figsize = [20,6*len(funcs)])
for i_f, fct in enumerate(funcs):
    Rt = Retrainer(D_G)
    if("period" in fct):
        getattr(Retrainer, fct)(Rt, width= 250, period=400)
    else:
        getattr(Retrainer, fct)(Rt)
    print(fct, Rt.detection_indices)
    Rt.plot_retrain(with_no_retrain=True, ax=ax[i_f])

# %%



