from ast import literal_eval
from src.bench_fcts import *
from src.retrain_fcts import *
from src.viz_fcts import *
from src.data_fcts import *
import os
import sys


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

# # Show df metrics

results_path = os.environ.get("RESULTS_ROOT_PATH")

single_drift = [f for f in list_df_generate_fcts if "nodrift" in f.__name__]
# %%

D_G = single_drift[-1]()#generate_abrupt_concept_drift(n_samples=5000)
print(D_G.drift_name)
print(results_path)
results = os.listdir(results_path)

all_df_metrics = get_D_G_metrics(D_G, results, selected_methods = ["ADWIN", "PH", "KSWIN"], exp_type="noisy", noisy = True)
metrics_list = ['no', 'false', 'min','mean', 'median', 'median_true', 'max', 'std']


for i,f in enumerate(single_drift[1:-1]):

    D_G = f()#generate_abrupt_concept_drift(n_samples=5000)
    print(D_G.drift_name)
    all_df_metrics = get_D_G_metrics(D_G, results, selected_methods = ["ADWIN", "PH", "KSWIN"], exp_type="noisy", noisy = True)
    if(i==0):
        df_drift_res = all_df_metrics
    else:
        df_drift_res = pd.concat([df_drift_res, all_df_metrics], axis=1)
    #all_df_metrics = all_df_metrics.loc[metrics_list].T.sort_values(["no","false","median"]).sort_index()
    
df_drift_res = df_drift_res.loc[metrics_list].T.sort_values(["no","false","median"]).sort_index()
df_drift_res = df_drift_res.drop(columns = ["no","false"])
df = df_drift_res.copy()
df['noise_rate'] = [x.split("_")[-1] for x in df.index]
df['algo'] = ["_".join(x.split("_")[1:-1]) for x in df.index]

D_G = single_drift[1]()#generate_abrupt_concept_drift(n_samples=5000)
print(D_G.drift_name)

all_df_metrics = get_D_G_metrics(D_G, results, selected_methods = ["ADWIN", "PH", "KSWIN"], exp_type="noisy", noisy = True)

all_df_metrics.loc[metrics_list].T.sort_values(["no","false","median"]).sort_index()


for i,f in enumerate(single_drift[1:-1]):

    D_G = f()#generate_abrupt_concept_drift(n_samples=5000)
    print(D_G.drift_name)
    all_df_metrics = get_D_G_metrics(D_G, results, selected_methods = ["ADWIN", "PH", "KSWIN"], exp_type="noisy", noisy = True)
    if(i==0):
        df_drift_res = all_df_metrics
    else:
        df_drift_res = pd.concat([df_drift_res, all_df_metrics], axis=1)
    #all_df_metrics = all_df_metrics.loc[metrics_list].T.sort_values(["no","false","median"]).sort_index()
    
df_drift_res = df_drift_res.loc[metrics_list].T.sort_values(["no","false","median"]).sort_index()
    

# %%
restrict_words = ["adwin"]#, 'KSWIN'], adwin, PH``
restrict_index =  np.ravel([[x for x in df_drift_res.index if word in x] for word in restrict_words])
df_drift_res.loc[restrict_index].sort_values(["no","false","median"])#.sort_index()



# %%
df = df_drift_res.loc[:].loc[:,["median", "std", "no","false"]]#
df['noise_rate'] = [x.split("_")[-1] for x in df.index]
df['algo'] = ["_".join(x.split("_")[1:-1]) for x in df.index]


# %%
fig, ax = plot_evolving_noise_rates(df)
save_path = os.environ.get("FIGURE_PATH")+"_".join(single_drift[0].__name__.split("_")[2:])
save_path += 'evolving_detect_rates.png'
print(save_path)
plt.savefig(save_path, bbox_inches='tight')
#plt.show()

# %%
#for d_name in ["nodrift"]:
#    single_drift = [f for f in list_df_generate_fcts if d_name in f.__name__]#[:3]
#
#    for i,f in enumerate(single_drift[1:]):
#
#        D_G = f()#generate_abrupt_concept_drift(n_samples=5000)
#        #print(D_G.drift_name)
#        all_df_metrics = get_D_G_metrics(D_G, results, selected_methods = ["ADWIN", "PH", "KSWIN"], exp_type="noisy", noisy = True);
#        if(i==0):
#            df_drift_res = all_df_metrics
#        else:
#            df_drift_res = pd.concat([df_drift_res, all_df_metrics], axis=1)
#        #all_df_metrics = all_df_metrics.loc[metrics_list].T.sort_values(["no","false","median"]).sort_index()
#    df_drift_res = df_drift_res.loc[metrics_list].T.sort_values(["no","false","median"]).sort_index()
#    df = df_drift_res.loc[:].loc[:,["median", "std", "no","false"]]#
#    df['noise_rate'] = [x.split("_")[-1] for x in df.index]
#    df['algo'] = ["_".join(x.split("_")[1:-1]) for x in df.index]
#
#    fig, ax = plot_evolving_noise_rates(df)
#
#    save_path = os.environ.get("FIGURE_PATH")+"_".join(single_drift[0].__name__.split("_")[2:])
#    save_path += 'evolving_detect_rates.png'
#    print(save_path)
#    plt.savefig(save_path, bbox_inches='tight')
#    #plt.show()

# %%
selected_methods = ["ADWIN", "PH", "KSWIN"]

# %%
for f in list_df_generate_fcts: 
    D_G = f()
    print("#"*20, f"{D_G.drift_name} {D_G.n}", "#"*20)


for f in list_df_generate_fcts[1:]: 
    D_G = f()
    print("#"*20, f"{D_G.drift_name} {D_G.n}", "#"*20)
    df_results_dataset, sep_bar_indexes = get_df_detections(D_G, results,
                                                            selected_methods, exp_type="df_reset", noisy=True)
    
    for col in  ["retrain_PH_loss","retrain_KSWIN_0","retrain_KSWIN.", "retrain_adwin_loss"]:
        for real_col in df_results_dataset.columns:
            if(col in real_col):
    #    if(col in df_results_dataset.columns):
                df_results_dataset = df_results_dataset.drop(columns=[real_col])

    ax = plot_violins(D_G, df_results_dataset, ax=None, sep_bar_indexes=sep_bar_indexes,
                            separate_true_false=False);
    #ax.set_xlim(800,1150)
    #plt.show()

# %%
D_G = single_drift[0]()
selected_methods = [["ADWIN"],["PH"],["KSWIN"]][2]


selected_methods = ["PH"]#['ADWIN', 'PH', 'KSWIN']

# %%
for f in list_df_generate_fcts:
    D_G = f()
    print(D_G.drift_name, "#"*20)

D_G = list_df_generate_fcts[2]()
print(D_G.drift_name, "#"*20)
df_results_dataset, sep_bar_indexes = get_df_detections(D_G, results,
                                                        selected_methods, exp_type="df_reset", noisy=True)

# %%
for col in  ["retrain_PH_loss","retrain_KSWIN_0","retrain_KSWIN.", "retrain_adwin_loss"]:
    for real_col in df_results_dataset.columns:
        if(col in real_col):
#    if(col in df_results_dataset.columns):
            df_results_dataset = df_results_dataset.drop(columns=[real_col])
            print(f"{real_col} dropped")

# %%
try:
    drop_methods = ["retrain_PH_loss","retrain_KSWIN", "retrain_adwin_loss"]
    df_results_dataset = df_results_dataset.drop(columns=drop_methods)
    sep_bar_indexes = [2,4]#[2,5]
except:
    try:
        drop_methods = ["retrain_PH_loss","retrain_KSWIN"]
        df_results_dataset = df_results_dataset.drop(columns=drop_methods)
        sep_bar_indexes = [2,4]#[2,5]
    except:
        print("ok")


# %%
df_results_dataset.columns = [x.replace("_loss","") for x in df_results_dataset.columns]

# %%
ax = plot_violins(D_G, df_results_dataset.iloc[:,:], ax=None, sep_bar_indexes=sep_bar_indexes, separate_true_false=False);
#ax.set_xlim(800,1150)


#plt.legend([])
save_path = os.environ.get("FIGURE_PATH")+D_G.drift_name.replace(" ","_")
save_path += 'noisy_'+selected_methods[0]+'_compare.png'
print(save_path)
plt.savefig(save_path, bbox_inches='tight')
#plt.show()
