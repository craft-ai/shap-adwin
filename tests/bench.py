# %% [markdown]
# # Imports
import unittest

# %%
import os, sys

module_path = os.path.abspath(os.path.join('../'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.data_fcts import *
from src.retrain_fcts import *

# %%
from src.bench_fcts import *
from ast import literal_eval

# %% [markdown]
# # Drifts data showcase

# %%
dataset_list = ['sine1_short.csv', 'stagger.csv']

# %%
n_samples = 3000
D_G = Drift_generators(n_samples = n_samples, n_features = 3)
D_G.smooth_concept_drift(n_drift=1)
D_G.plot_drift();

# %%
D_G = Drift_generators(n_samples = n_samples, n_features = 3)
D_G.abrupt_concept_drift(n_drift=1)
D_G.plot_drift();

# %%
D_G = Drift_generators()
df = D_G.load_df("sine1_short")
D_G.plot_drift();

# %%
D_G = Drift_generators()
df = D_G.load_df("stagger_short")
D_G.plot_drift();

# %% [markdown]
# # list all D_G

# %%


# %%
n_samples = 3000
all_dg = []
D_G = Drift_generators(n_samples = n_samples, n_features = 3)
D_G.smooth_concept_drift(n_drift=1)
all_dg.append(D_G)
D_G = Drift_generators(n_samples = n_samples, n_features = 3)
D_G.abrupt_concept_drift(n_drift=1)
all_dg.append(D_G)
D_G = Drift_generators(n_samples = n_samples, n_features = 3)
d_centers = [(0, 0,
 0.3, 0.25, 1),(0, 0.3, 0.5, 0, 0.25),(0, 0.5, 1, 0.25, 1)]
D_G.abrupt_covariate_drift(d_centers=d_centers)
all_dg.append(D_G)
D_G = Drift_generators()
df = D_G.load_df("sine1_short")
all_dg.append(D_G)
D_G = Drift_generators()
df = D_G.load_df("stagger_short")
all_dg.append(D_G)

all_dg

# %%
funcs = [func for func in dir(Retrainer) if callable(getattr(Retrainer, func))
         and not func.startswith("__")
         and "retrain" in func and not "plot" in func and not "perf" in func]
[x for x in funcs if "shap" not in x]

# %%
adwin_funcs = ['retrain_adwin',
                'retrain_adwin_loss',
                'retrain_adwin_point',
                'retrain_adwin_shap',]
kswin_funcs = ['retrain_KSWIN',
                'retrain_KSWIN_loss',
                'retrain_KSWIN_point',
                'retrain_KSWIN_shap']

# %%
funcs = ['retrain_DDM',
         'retrain_EDDM',
         'retrain_HDDM_A',
         'retrain_KSWIN',
         'retrain_PH',
         'retrain_adwin']

# %%
#funcs = ['retrain_adwin', "retrain_PH", "retrain_ttest"]
#funcs = ['retrain_adwin',  "retrain_PH", "retrain_alibi_detector", "retrain_shap_adwin", "retrain_ks"]

# %%


# %%
all_results = []
for D_G in all_dg[:1]:
    print(D_G.drift_name)
    #D_G.plot_drift();
    results_D_G = []
    fig, ax = plt.subplots(nrows=1, ncols=len(funcs), figsize=[8*len(funcs), 10])
    for i, fct in enumerate(funcs):
        Rt = Retrainer(D_G)
        getattr(Retrainer, fct)(Rt)
        Rt.plot_retrain(with_no_retrain=True, ax=ax[i]);
        results_D_G.append(Rt.detection_indices)
    all_results.append(results_D_G)

# %%
#for i, D_G in enumerate(all_dg[:1]):
#    print(D_G.drift_name)
#    for res, fct in zip(all_results[i], funcs):
#        print(f"    {fct}: {res}")

# %%


# %% [markdown]
# # tests

# %%
def generate_new_df():
    
    D_G = Drift_generators()
    D_G.load_df("stagger_short")
    #D_G = Drift_generators(n_samples = 5000, n_features = 3)
    #D_G.abrupt_concept_drift(n_drift=1)
    return(D_G)

# %%
D_G = generate_new_df()

# %%


# %% [markdown]
# ## Test PH

# %%
Rt = Retrainer(D_G)
#Rt.retrain_PH()
for delta in []:
    print()
Rt.retrain_PH(delta_d = 0.1, stop_first_detect=True)
#Rt.retrain_PH_loss(alpha=0.0001, w_size=D_G.n_train, stat_size=D_G.n_test)
Rt.plot_retrain();

# %%
Rt = Retrainer(D_G)
#Rt.retrain_PH()
#Rt.retrain_PH_shap()
Rt.retrain_PH_loss(delta_d=2, stop_first_detect =True)
Rt.plot_retrain();

# %%
Rt = Retrainer(D_G)
#Rt.retrain_PH()
#Rt.retrain_PH_shap()
Rt.retrain_PH_shap(delta_d=1.5, stop_first_detect =True)
Rt.plot_retrain();

# %% [markdown]
# ## Test KSWIN

# %%
Rt = Retrainer(D_G)
#Rt.retrain_KSWIN()
#Rt.retrain_KSWIN_shap()
Rt.retrain_KSWIN(alpha=0.0001, w_size=D_G.n_train, stat_size=D_G.n_test, stop_first_detect =True)
Rt.plot_retrain();

# %%
Rt = Retrainer(D_G)
#Rt.retrain_KSWIN()
#Rt.retrain_KSWIN_shap()
Rt.retrain_KSWIN_loss(alpha=0.0001, w_size=D_G.n_train, stat_size=D_G.n_test, stop_first_detect =True)
Rt.plot_retrain();

# %%
Rt = Retrainer(D_G)
#Rt.retrain_KSWIN()
Rt.retrain_KSWIN_shap(alpha=0.0001, w_size=D_G.n_train, stat_size=D_G.n_test, stop_first_detect =True)
#Rt.retrain_KSWIN_loss()
Rt.plot_retrain(with_no_retrain=True);

# %%
params_retrain = {"alpha":0.0001, "w_size":D_G.n_train, "stat_size":D_G.n_test, "stop_first_detect" :True}
funcs = ["retrain_KSWIN_shap", "retrain_KSWIN_loss", "retrain_KSWIN"]


# %%
def generate_new_df():
    D_G = Drift_generators(n_samples = 3000, n_features = 3)
    D_G.smooth_concept_drift(n_drift=1)
    return(D_G)

# %%
df_results = pd.read_csv("../data/results/abrupt_concept_drift_5000_df_reset_ADWIN.csv", index_col=0)
for col in df_results.columns:
    df_results.loc[:,col] = df_results.loc[:,col].apply(literal_eval)
df_results

# %%
D_G = generate_new_df()
D_G.drifts[0].start

# %%
get_df_metrics(df_results, D_G.drifts[0].start)

# %%


# %% [markdown]
# # Generate results

# %%
kswin_funcs

# %%
adwin_funcs = ['retrain_adwin',"retrain_adwin_shap"]
KSWIN_funcs = ['retrain_KSWIN',"retrain_KSWIN_shap"]
PH_funcs = ['retrain_PH', "retrain_PH_loss", "retrain_PH_shap"]

# %%

D_G = Drift_generators()
D_G.load_df("stagger_short")

# %%
D_G.plot_drift()

# %%


# %%
def generate_new_df():
#    D_G = Drift_generators(n_samples = 3000, n_features = 3)
#    D_G.smooth_concept_drift(n_drift=1)

    D_G = Drift_generators()
    D_G.load_df("stagger_short")
    return(D_G)

# %%
loss_vs_funcs = ['retrain_KSWIN',
                 'retrain_KSWIN_loss',
                 'retrain_PH',
                 'retrain_PH_loss',
                 'retrain_adwin',
                 'retrain_adwin_loss',
 ]

# %%
adwin_funcs

# %%
def generate_new_df():
    
    D_G = Drift_generators(n_samples = 5000, n_features = 3)
    D_G.abrupt_concept_drift(n_drift=1)
    return(D_G)

# %%
n_iter = 10

D_G = generate_new_df()
D_G.drift_name

# %%
kswin_funcs = ['retrain_KSWIN',
                'retrain_KSWIN_loss',
                'retrain_KSWIN_shap']

# %%
PH_funcs = ['retrain_PH',
                'retrain_PH_loss',
                'retrain_PH_shap']

# %%
params_retrain = {"alpha":0.0001, "w_size":D_G.n_train, "stat_size":D_G.n_test, "stop_first_detect" :True}
funcs = ["retrain_KSWIN_shap", "retrain_KSWIN_loss", "retrain_KSWIN"]

# %%

results_root_path = os.environ.get("RESULTS_ROOT_PATH")
def eval_DG_model_save(generate_new_df, funcs, params_retrain, expe_set_name, n_iter):
    
    D_G = generate_new_df()
    results_df = eval_n_time_D_G(generate_new_df, funcs, params_retrain=params_retrain, n_iter=n_iter)
    results_path = results_root_path+D_G.drift_name.replace(" ","_")+f"_{len(D_G.df)}_df_reset_{expe_set_name}.csv"
    results_df.to_csv(results_path)
    
    results_df_model = eval_n_time_model(D_G, funcs, params_retrain=params_retrain, n_iter=n_iter, shuffle=True)
    results_path = results_root_path+D_G.drift_name.replace(" ","_")+f"_{len(D_G.df)}_model_reset_{expe_set_name}.csv"
    results_df_model.to_csv(results_path)
    return(results_df_model, results_df)

# %%
funcs

# %%
a,b = eval_DG_model_save(generate_new_df, funcs, params_retrain, "KSWIN", n_iter=2)

# %%


# %%
b

# %%
funcs = ["retrain_KSWIN_shap", "retrain_KSWIN_loss", "retrain_KSWIN"]
params_retrain = {f:{"alpha":0.0001, "w_size":D_G.n_train, "stat_size":D_G.n_test, "stop_first_detect" :True}
                     for f in funcs}

# %%
results_df = eval_n_time_D_G(generate_new_df, funcs, params_retrain=params_retrain, n_iter=10)
results_df

# %%
#results_df = eval_n_time_D_G(generate_new_df, PH_funcs, n_iter=n_iter)

# %%
#results_path = "/home/bastien/Documents/labs/data/results/"+D_G.drift_name.replace(" ","_")+"_df_reset.csv"
expe_set_name = 'KSWIN'
results_path = os.environ.get("RESULTS_ROOT_PATH")+D_G.drift_name.replace(" ","_")+f"_{len(D_G.df)}_df_reset_{expe_set_name}.csv"
results_df.to_csv(results_path)
print(results_path)

# %%
results_df

# %%
D_G = generate_new_df()
df_metrics = get_df_metrics(results_df, drift_start=D_G.drifts[0].start)
df_metrics

# %%
params_retrain

# %%
D_G = generate_new_df()
results_df_model = eval_n_time_model(D_G, funcs, params_retrain=params_retrain, n_iter=10, shuffle=True)

# %%
#results_path = "/home/bastien/Documents/labs/data/results/"+D_G.drift_name.replace(" ","_")+"_df_reset.csv"
results_path = os.environ.get("RESULTS_ROOT_PATH")+D_G.drift_name.replace(" ","_")+f"_{len(D_G.df)}_model_reset_{expe_set_name}.csv"
results_df_model.to_csv(results_path)
print(results_path)

# %%
results_df_model

# %%
df_metrics = get_df_metrics(results_df_model, drift_start=D_G.drifts[0].start)
df_metrics

# %%


if __name__ == "__main__":
     unittest.main()

