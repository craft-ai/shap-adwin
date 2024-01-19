# %%


# %%
import numpy
numpy.__version__

# %%
import cython
cython.__version__

# %%
import river.drift


# %%
import os, sys

module_path = os.path.abspath(os.path.join('../'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.data_fcts import *
from src.retrain_fcts import *
from src.detectors_fcts import Ttest_detector

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%


# %% [markdown]
# # river time series

# %%
from statsmodels.tsa.arima.model import ARIMA

# %%


# %%
mod = ARIMA(ts, order=(1, 0, 0))
res = mod.fit()
print(res.summary())


# %%


# %%


# %%


# %% [markdown]
# # model online

# %%
df = load_river_data(dataset_name="insects", n_samples=10)

# %%
df

# %%


# %%


# %% [markdown]
# # TEST load data

# %%
dir(river.datasets)[:10]

# %%
from river.datasets import *

# %%
data = Elec2()
data.download()

# %%
exec(f"from river.datasets import Elec2 as riverDataset")

# %%
riverDataset()

# %%
def load_river_data(dataset_name="Elec2", n_samples=None):
    exec(f"from river.datasets import {dataset_name} as riverDataset")
    data = riverDataset()
    data.download()
    df = pd.DataFrame(columns = list([x for x in data.take(1)][0][0].keys()))
    label = []
    if n_samples is None: 
        n_samples = data.n_samples
    for i, x in enumerate(data.take(n_samples)):
        df.loc[i] = pd.Series(x[0])
        label.append(x[1])
    return(df, label)

# %%
df, label = load_river_data(dataset_name="Elec2", n_samples=10)

# %%
data.task, data.n_features, data.n_samples, data.n_classes, data.n_outputs, data.sparse

# %%
df = pd.DataFrame(columns = list([x for x in data.take(1)][0][0].keys()))
label = []
#for i, x in enumerate(data.take(data.n_samples)):
for i, x in enumerate(data.take(5000)):
    df.loc[i] = pd.Series(x[0])
    label.append(x[1])
    

# %%
plt.plot(label[:100])

# %% [markdown]
# # Test implement custom detector

# %%
from river.base import DriftDetector

# %%
D_G = Drift_generators(n_samples = 1000, n_features = 3)
D_G.abrupt_concept_drift(n_drift=1, drifts=[Drift(True, 400,None, [0.7, 0.7, 0.7])])
#D_G.smooth_concept_drift(n_drift=1)
Rt = Retrainer(D_G)
Rt.plot_retrain(w_size = 100)

# %%
import numbers
from river.base import DriftAndWarningDetector

# %%

from river.drift import ADWIN, KSWIN, PageHinkley

# %%


# %%
D_G = Drift_generators(n_samples = int(5e2), n_features = 3)
d_centers = [(0, 0,
 0.3, 0.25, 1),(0, 0.3, 0.5, 0, 0.25),(0, 0.5, 1, 0.25, 1)]
D_G.abrupt_covariate_drift(d_centers=d_centers)
Rt = Retrainer(D_G)
Rt.plot_retrain(w_size = 50)

# %%
a = Ttest_detector(**{"Rt":Rt, "win_size":5, "clock":1, "p_val_threshold":1e-2})


# %%
a.ref_win[0] == Rt.df.drop(columns=['class']).iloc[Rt.n_train:Rt.n_train+a.win_size,0].values

# %%
a.ref_win

# %%
a.win[0] == a.ref_win[0]#Rt.X_unseen.values[10-5:10,0]

# %%
i =0 
for index, (x, label) in enumerate(zip(Rt.X_unseen.values, Rt.y_unseen.values)):
    if(i>=10000):
        break
    a.update(x);
    if(a.drift_detected):
        for f in range(Rt.n_features):
            _, p_val = ttest_ind( a.ref_win[f], a.win[f])
            print(p_val, a.win[0] == a.ref_win[0])
        print("DONE")
        a._reset()
    i+=1

# %%

Rt = Retrainer(D_G)
#ttestdetector = Ttest_detector(Rt=Rt)

Rt.retrain_detector(Ttest_detector, detector_params={"Rt":Rt, "win_size":300, "clock":10, "p_val_threshold":1e-3}, signal='point', warning_inside=True);
#Rt.retrain_adwin(delta_d=0.99);

# %%
a.ref_win[0] == a.win[0]

# %%

Rt = Retrainer(D_G)
#ttestdetector = Ttest_detector(Rt=Rt)

Rt.retrain_detector(Ttest_detector, detector_params={"Rt":Rt, "win_size":40, "clock":1, "p_val_threshold":1e-2}, signal='point', warning_inside=True)
#Rt.retrain_adwin(delta_d=0.99);
Rt.detection_indices

# %%

Rt = Retrainer(D_G)
#ttestdetector = Ttest_detector(Rt=Rt)

Rt.retrain_ttest(win_size= 40, clock=1, p_val_threshold=1e-2)
#Rt.retrain_adwin(delta_d=0.99);

Rt.detection_indices

# %%
Rt.plot_retrain()

# %%
win_size=30
a= [list(Rt.X_unseen.iloc[:win_size, feat].values)
                            for feat in range(Rt.n_features)]
b= a.copy()
b[0]                            

# %%
print(Rt.detection_indices)
Rt.plot_retrain()


# %%
strat=200
feat = 0
n=30

a,b = ttest_ind(Rt.df.drop(columns=['class']).iloc[Rt.n_train:Rt.n_train+n, feat].values, Rt.df.drop(columns=['class']).iloc[strat:strat+n, feat].values)
a, b

# %%
Rt = Retrainer(D_G)

# %%
n=200
ttest_ind(np.arange(n), np.arange(n))

# %%


# %%
Retrainer(D_G).plot_retrain()

# %%
strat=4000
feat = 2

a,b = ttest_ind(Rt.df.drop(columns=['class']).iloc[Rt.n_train:Rt.n_train+100, feat].values, Rt.X_unseen.iloc[strat:strat+100, feat].values)
a, b

# %%
Rt.plot_retrain()

# %%

Rt = Retrainer(D_G)
#ttestdetector = Ttest_detector(Rt=Rt)
Rt.retrain_adwin();#retrain_detector(Ttest_detector, detector_params={"Rt":Rt, "win_size":50}, signal='point')

# %%
D_G = Drift_generators(n_samples = 5000, n_features = 3)
D_G.abrupt_concept_drift(n_drift=1, drifts=[Drift(True, 2000,None, [0.7, 0.7, 0.7])])
#D_G.smooth_concept_drift(n_drift=1)
Rt = Retrainer(D_G)
#Rt.plot_retrain(w_size = Rt.n_test)

# %%
win_size=200
ttestdetector = Ttest_detector(Rt=Rt)
a
#ttest_detector.consec_feat_detections = [0 for x in range(D_G.n_features)]
#ttest_detector.consec_detect = [0 for x in range(4)]
#ttest_detector.ref_win = [Rt.X_unseen.iloc[:win_size, feat]
#                           for feat in range(Rt.n_features)]

#def ttest_update(self, point):

#ttest_detector.consec

# %%
Rt.plot_retrain()

# %% [markdown]
# # Test lib

# %%
import random
from river import drift
rng = random.Random(12345)
detectors = [drift.PageHinkley(), drift.ADWIN(), drift.KSWIN()]
#detectors = [ drift.KSWIN()]
# Simulate a data stream composed by two data distributions
data_stream = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)
# Update drift detector and verify if change is detected
for i, val in enumerate(data_stream):
    for d in detectors:
        _ = d.update(val)
        if d.drift_detected:
            print(f"Change detected at index {i}, input value: {val}, {d}")
            

# %%
import random
from river import drift
rng = random.Random(42)

bin_detectors = [drift.binary.DDM(), drift.binary.EDDM(), drift.binary.HDDM_A(), drift.binary.HDDM_W()]
# Simulate a data stream where the first 1000 instances come from a uniform distribution
# of 1's and 0's
data_stream = rng.choices([0, 1], k=1000)
# Increase the probability of 1's appearing in the next 1000 instances
data_stream = data_stream + rng.choices([0, 1], k=1000, weights=[0.3, 0.7])
print_warning = True
# Update drift detector and verify if change is detected
for i, x in enumerate(data_stream):
     for d in bin_detectors:
        _ = d.update(x)
        #if d.warning_detected and print_warning:
        #    print(f"Warning detected at index {i} - {d}")
        #    print_warning = False
        if d.drift_detected:
            print(f"Change detected at index {i} - {d}")
            print_warning = True

# %% [markdown]
# 

# %%
D_G = Drift_generators(n_samples = 5000, n_features = 3)
D_G.abrupt_concept_drift(n_drift=1, drifts=[Drift(True, 2000,None, [0.7, 0.7, 0.7])])
#D_G.smooth_concept_drift(n_drift=1)

D_G.train()
Rt = Retrainer(D_G)
Rt.plot_retrain(w_size = Rt.n_test)

# %%
Rt = Retrainer(D_G)
_, detection_indices = Rt.retrain_detector(detector_func = drift.ADWIN, warning_inside=False,
                                                detector_params={"delta":0.01, "clock":1});
print(detection_indices )
Rt.plot_retrain(with_no_retrain=True)#w_size = Rt.n_test

# %%
Rt = Retrainer(D_G)
_, detection_indices = Rt.retrain_detector(detector_func = drift.ADWIN, warning_inside=False,
                                                detector_params={"delta":0.05, "clock":1});
print(detection_indices )
Rt.plot_retrain(with_no_retrain=True)#w_size = Rt.n_test

# %%
Rt.compute_errors();

# %%
Rt.plot_retrain()

# %%
Rt = Retrainer(D_G)
_, detection_indices = Rt.retrain_detector(detector_func = drift.ADWIN, warning_inside=False,
                                                detector_params={"delta":0.01, "clock":1});
print(detection_indices )
Rt.plot_retrain()#w_size = Rt.n_test

# %%

from river import drift

# %%
for delta in [0.5, 0.1, 0.01, 0.05, 0.002]:

    Rt = Retrainer(D_G)
    _, detection_indices = Rt.retrain_detector(detector_func = drift.ADWIN, warning_inside=False,
                                                    detector_params={"delta":delta, "clock":1});
    print(f"delta: {delta} -> {detection_indices}")
    Rt = Retrainer(D_G)
    
    _, detection_indices = Rt.retrain_adwin(delta_d=delta);
    print(f"               -> {detection_indices}")

# %%
for delta in [0.5, 0.1, 0.01, 0.05, 0.001, 0.005, 0.0001]:

    Rt = Retrainer(D_G)
    _, detection_indices = Rt.retrain_detector(detector_func = drift.PageHinkley, warning_inside=False,
                                                    detector_params={"delta":delta});
    print(f"delta: {delta} -> {detection_indices}")
    Rt = Retrainer(D_G)
    _, detection_indices = Rt.retrain_PH(delta_d=delta);
    print(f"               -> {detection_indices}")

# %%
for d in [drift.PageHinkley, drift.ADWIN, drift.KSWIN]:
    Rt = Retrainer(D_G)
    errors, detection_indices = Rt.retrain_detector(detector_func = d);
    print(detection_indices, d)

# %%
for d in [drift.binary.DDM, drift.binary.EDDM, drift.binary.HDDM_A, drift.binary.HDDM_W]:
    Rt = Retrainer(D_G)
    errors, detection_indices = Rt.retrain_detector(detector_func = d, warning_inside=True, signal="error");
    print(detection_indices, d)

# %% [markdown]
# 

# %%



