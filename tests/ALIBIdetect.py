import numpy
numpy.__version__
import os, sys

module_path = os.path.abspath(os.path.join('../'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.data_fcts import *
from src.retrain_fcts import *
from src.detectors_fcts import *
import matplotlib.pyplot as plt
import seaborn as sns
D_G = Drift_generators(n_samples = 1000, n_features = 3)
D_G.abrupt_concept_drift(n_drift=1, drifts=[Drift(True, 400,None, [0.7, 0.7, 0.7])])
#D_G.smooth_concept_drift(n_drift=1)
Rt = Retrainer(D_G)
Rt.plot_retrain(w_size = 100)


from alibi_detect.cd import KSDrift, RegressorUncertaintyDrift

from alibi_detect.cd import MMDDrift

x_ref=np.array(np.ones((3,4))).T

MMDDrift(x_ref=x_ref)
win_size=200
adetector = Alibi_generic_detector(Rt=Rt, feat_wise=False,clock=60, alibi_detector=MMDDrift)

for index, (x, label) in enumerate(zip(Rt.X_unseen.values, Rt.y_unseen.values)):
    #error = int(self.model.predict([x]) == label)
    #errors.append(error)
    adetector.update(x)
    break
win_size=200
adetector = Alibi_generic_detector(Rt=Rt, feat_wise=False,clock=10, alibi_detector=MMDDrift)

for index, (x, label) in enumerate(zip(Rt.X_unseen.values, Rt.y_unseen.values)):
    #error = int(self.model.predict([x]) == label)
    #errors.append(error)
    adetector.update(x)
    if(adetector.drift_detected):
        break
print(index)
Ttest_detector
D_G = Drift_generators(n_samples = int(1e3), n_features = 3)
d_centers = [(0, 0,
 0.3, 0.25, 1),(0, 0.3, 0.5, 0, 0.25),(0, 0.5, 1, 0.25, 1)]
D_G.abrupt_covariate_drift(d_centers=d_centers)
Rt = Retrainer(D_G)
Rt.plot_retrain()

Rt = Retrainer(D_G)
ref_win = [list(Rt.df.drop(columns=['class']).iloc[Rt.n_train:Rt.n_train+min(win_size, Rt.n_test),
                                                              feat].values)
                   for feat in range(Rt.n_features)]
Rt.retrain_detector(detector_func=Ttest_detector,
                                signal='point',
                                detector_params={"ref_win": ref_win,
                                                "win_size": win_size,
                                                "clock": 1,
                                                "p_val_threshold": 1e-2},
                                warning_inside=True,
                                retrain_name="ks");
Rt.detection_indices
from alibi_detect.cd import LSDDDrift, MMDDrift, ContextMMDDrift
#LearnedKernelDrift -> requires kernel argument
#ContextMMDDrift -> requires c_ref argument
#SpotTheDiffDrift -> error strange 
from alibi_detect.cd import ChiSquareDrift, KSDrift, CVMDrift, FETDrift
#FETDrift -> ValueError: `alternative` must be either 'greater', 'less' or 'two-sided'
Rt = Retrainer(D_G)
ref_win = [list(Rt.df.drop(columns=['class']).iloc[Rt.n_train:Rt.n_train+min(win_size, Rt.n_test),
                                                              feat].values)
                   for feat in range(Rt.n_features)]
Rt.retrain_detector(detector_func=Alibi_generic_detector,
                                signal='point',
                                detector_params={"alibi_detector":CVMDrift,#Alibi_generic_detector
                                                 "feat_wise": True,
                                                 "ref_win": ref_win,
                                                 "win_size": win_size,
                                                 "n_consec_w":1,
                                                 "n_consec_d":3,
                                                 "clock": 10,
                                                 "p_val_threshold": 1e-9},
                                warning_inside=True,
                                retrain_name="CVMDrift");
Rt.detection_indices
Rt.plot_retrain(with_no_retrain=True)
Rt = Retrainer(D_G)
ref_win = [list(Rt.df.drop(columns=['class']).iloc[Rt.n_train:Rt.n_train+min(win_size, Rt.n_test),
                                                              feat].values)
                   for feat in range(Rt.n_features)]
Rt.retrain_detector(detector_func=Alibi_generic_detector,
                                signal='point',
                                detector_params={"alibi_detector":KSDrift,#Alibi_generic_detector
                                                 "feat_wise": True,
                                                "ref_win": ref_win,
                                                "win_size": win_size,
                                                "n_consec_w":10,
                                                "n_consec_d":3,
                                                "clock": 1,
                                                "p_val_threshold": 1e-9},
                                warning_inside=True,
                                retrain_name="ks");
Rt.detection_indices
Rt.plot_retrain(with_no_retrain=True)
for detector in [LSDDDrift, MMDDrift, ChiSquareDrift, KSDrift, CVMDrift]:#ContextMMDDrift
    print(detector.__name__)
    Rt = Retrainer(D_G)
    Rt.retrain_alibi_detector(p_val_threshold=1e-9, alibi_detector=detector, 
                            win_size=win_size, n_consec_w=10, n_consec_d=3);
    print(Rt.detection_indices)
Rt = Retrainer(D_G)
Rt.retrain_alibi_detector(p_val_threshold=1e-9, alibi_detector=CVMDrift, 
                           win_size=win_size, n_consec_w=10, n_consec_d=3);
Rt.detection_indices
Rt.plot_retrain(with_no_retrain=True)
