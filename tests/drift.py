# %%
import os, sys

module_path = os.path.abspath(os.path.join('../'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.data_fcts import *

# %%

import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown]
# # Test add noise

# %%
#X, y = D_G.get_x_y(ind_start=0, ind_end=D_G.n)

# %%
n_samples = 100
D_G = Drift_generators(n_samples = n_samples, n_features = 3)
d_centers = [(0, 0,
            0.3, 0.25, 1),(0, 0.3, 0.5, 0, 0.25),(0, 0.5, 1, 0.25, 1)]
D_G.abrupt_covariate_drift(d_centers=d_centers)
D_G.add_noise(noise_rate=0.05)

# %%
X, y = D_G.get_x_y(ind_start=0, ind_end=D_G.n)
noisy_inds = y.sample(int(D_G.n*0.1)).index
#D_G.df.loc[noisy_inds, D_G.objective_col] = y[noisy_inds].apply(lambda x: np.random.choice([a for a in y.unique() if a != x]))

# %%
list(y[noisy_inds].apply(lambda x: np.random.choice([a for a in y.unique() if a != x])).values)

# %%
D_G.df.loc[noisy_inds, D_G.objective_col] = list(y[noisy_inds].apply(lambda x: np.random.choice([a for a in y.unique() if a != x])).values)

# %%
n_samples = 5000
D_G = Drift_generators(n_samples = n_samples, n_features = 3)

D_G.gradual_concept_drift(n_drift=1)
#D_G.add_noise(noise_rate=)
D_G.plot_drift(sample_frac= 1);

# %%
D_G.drift_name

# %%

D_G.add_noise(noise_rate=0.001)
D_G.plot_drift(sample_frac= 1);

# %%
D_G.drift_name

# %%


# %%


# %% [markdown]
# # Test Gradual concept Drift

# %%
n_samples = 5000
D_G = Drift_generators(n_samples = n_samples, n_features = 3)

D_G.gradual_concept_drift(n_drift=1)
D_G.plot_drift(sample_frac= 1);

# %%
D_G.drifts

# %% [markdown]
# # Test Smooth Concept Drift

# %%
n_samples = 10000
D_G = Drift_generators(n_samples = n_samples, n_features = 3)
D_G.smooth_concept_drift(n_drift=6)
D_G.plot_drift();

# %%
D_G.drifts

# %%
D_G = Drift_generators(n_samples = n_samples, n_features = 2)
D_G.smooth_concept_drift(n_drift=1)
D_G.plot_circles()

# %% [markdown]
# # Abrupt Concept Drift

# %%
n_samples = 10000
D_G = Drift_generators(n_samples = n_samples, n_features = 3)
D_G.abrupt_concept_drift(n_drift=6)
D_G.plot_drift();

# %%
D_G.plot_circles()

# %% [markdown]
# # Abrupt covariate Drift

# %%
D_G = Drift_generators(n_samples = int(10**4), n_features = 3)

d_centers = [(0, 0, 0.3, 0.25, 1),(0, 0.3, 0.5, 0, 0.25),(0, 0.5, 1, 0.25, 1)]
D_G.abrupt_covariate_drift(d_centers=d_centers)

# %%
D_G.drifts,

# %%
D_G.plot_drift();

# %%
D_G.drifts

# %% [markdown]
# brutal_concept_drift
# brutal_covariate_drift
# new_smooth_concept_drift
# smooth_concept_drift#Seems not relevant anymore
# 
# cyclic_abrupt_concept_drift
# back_and_forth_abrupt_drift
# back_and_forth_smooth_drift
# #add_noise

# %%
feat_cols = [0,1,2]

# %%
n_samples = 1000
D_G = Drift_generators(n_samples = n_samples, n_features = 3)
D_G.cyclic_abrupt_concept_drift(n_drift=2)#, drift_points=[int(n_samples*1/2)], 
        #circle_centers=[[0.4 for i in range(D_G.n_features)],[0.6 for i in range(D_G.n_features)]])# = None, circle_centers = None):

df = D_G.df

fig, ax = plt.subplots(nrows=2, ncols=len(df.columns)-1, figsize= [20,5])
for class_val in range(2):
    bef = df[df.loc[:,"class"]==class_val].loc[0:D_G.drift_points[0], feat_cols]#before Drift
    aft = df[df.loc[:,"class"]==class_val].loc[D_G.drift_points[0]:, feat_cols]#After Drift
    for feat in range(len(df.columns)-1):
        sns.regplot(x=np.arange(len(bef)), y=bef.loc[:,feat], label='Before', ax =ax[class_val, feat])
        sns.regplot(x=np.arange(len(aft)), y=aft.loc[:,feat], label='After', ax =ax[class_val, feat])
        ax[class_val, feat].legend()
        ax[class_val, feat].set_title(f"col : {feat}")
        ax[class_val, feat].set_xlabel("positive class instances")
        ax[class_val, feat].set_ylabel(f"feat:{feat} value")
plt.suptitle(f"Drift_points: {D_G.drift_points}")
plt.show()


# %%
n_samples = 1000
D_G = Drift_generators(n_samples = n_samples, n_features = 3)
D_G.cyclic_abrupt_concept_drift(n_drift=3)#, drift_points=[int(n_samples*1/2)], 
        #circle_centers=[[0.4 for i in range(D_G.n_features)],[0.6 for i in range(D_G.n_features)]])# = None, circle_centers = None):

df = D_G.df

fig, ax = plt.subplots(nrows=2, ncols=len(df.columns)-1, figsize= [20,5])
for class_val in range(2):
    bef = df[df.loc[:,"class"]==class_val].loc[0:D_G.drift_points[0], feat_cols]#before Drift
    aft = df[df.loc[:,"class"]==class_val].loc[D_G.drift_points[0]:, feat_cols]#After Drift
    for feat in range(len(df.columns)-1):
        sns.regplot(x=np.arange(len(bef)), y=bef.loc[:,feat], label='Before', ax =ax[class_val, feat])
        sns.regplot(x=np.arange(len(aft)), y=aft.loc[:,feat], label='After', ax =ax[class_val, feat])
        ax[class_val, feat].legend()
        ax[class_val, feat].set_title(f"col : {feat}")
        ax[class_val, feat].set_xlabel("positive class instances")
        ax[class_val, feat].set_ylabel(f"feat:{feat} value")
plt.suptitle(f"Drift_points: {D_G.drift_points}")
plt.show()


# %%



