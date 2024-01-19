
from river.base import DriftAndWarningDetector
from scipy.stats import ttest_ind
import numpy as np
from alibi_detect.cd import KSDrift


class Ttest_detector():
    """
    Data drift detector, raises a detection when the Ttest of a given window
    is significantly different with the reference window

    The larger the window the more reliable the test but the later the detection
    as the window distribution takes time to adapt itself


    """

    def __init__(self, p_val_threshold=1e-10, ref_win=None, win_size=50, Rt=None, n_consec_d=6, n_consec_w=1, clock=1):
        self.drift_detected = False
        self.warning_detected = False
        self.p_val_threshold = p_val_threshold
        if(ref_win is None and Rt is None):
            print("PLEASE PROVIDE REF WIN OR BASE RETRAINER")
        elif(ref_win is not None):
            self.consec_feat_detections = [0 for x in range(len(ref_win))]
            self.ref_win = ref_win
            self.win = ref_win
            self.win_size = len(ref_win[0])
        elif(Rt is not None):
            self.win_size = win_size
            self.consec_feat_detections = [0 for x in range(Rt.n_features)]
            self.ref_win = [list(Rt.df.drop(columns=['class']).iloc[Rt.n_train:Rt.n_train+min(win_size, Rt.n_test), feat].values)
                            for feat in range(Rt.n_features)]
            self.win = [list(Rt.df.drop(columns=['class']).iloc[Rt.n_train:Rt.n_train+min(win_size, Rt.n_test), feat].values)
                        for feat in range(Rt.n_features)]
        if(len(self.win[0]) == self.win_size):
            self.window_full = True
        else:
            self.window_full = False
        self.n_consec_d = n_consec_d
        self.n_consec_w = n_consec_w
        self.clock = clock
        self.counter = 0
        self.p_val = None

    def _reset(self):
        # This is a semi-hard-_reset, nothing is kept in win
        # other alternatives can be explored.
        self.drift_detected = False
        self.warning_detected = False
        self.consec_feat_detections = [0 for x in range(len(self.ref_win))]
        self.ref_win = self.win
        self.win = [[self.win[f][-1]] for f in range(len(self.ref_win))]
        self.window_full = False  # [Rt.X_unseen.iloc[:win_size, feat]
        # for feat in range(Rt.n_features)]  # TODO change ref_window based on last_values

    def update(self, x):  # numbers.Number) -> "DriftDetector":#TODO update with correct types
        self.counter += 1
        for feat, x_f in enumerate(x):
            self.win[feat].append(x_f)
            if(self.window_full):
                del self.win[feat][0]

            if(self.counter % self.clock == 0):
                # First return object is t-stat
                _, p_val = ttest_ind(self.win[feat], self.ref_win[feat])
                if(p_val < self.p_val_threshold):
                    self.p_val = p_val
                    self.consec_feat_detections[feat] += 1
                    if(self.consec_feat_detections[feat] > self.n_consec_w):
                        self.warning_detected = True
                        if(self.consec_feat_detections[feat] > self.n_consec_d):
                            self.drift_detected = True

                else:
                    self.consec_feat_detections[feat] = 0

        if(not self.window_full):
            # only active while window not full
            if(len(self.win[feat]) == self.win_size):
                self.window_full = True

    def __repr__(self):
        return(f"TTest Drift detector, current window has {len(self.win[0])} elts, and the detector has seen {self.counter} points {self.p_val }")

    def drift_detected(self):
        return self.drift_detected

    def warning_detected(self):
        return self._warning_detected


# -------------------------------------------------------------------------------------------------------------------------------------------------
class Alibi_generic_detector():
    """`
    TODO: Add description


    """

    def __init__(self, p_val_threshold=1e-10, ref_win=None, win_size=50, Rt=None, n_consec_d=6,
                 n_consec_w=1, clock=1,
                 alibi_detector=KSDrift,
                 feat_wise=True):
        self.drift_detected = False
        self.warning_detected = False
        self.p_val_threshold = p_val_threshold
        self.feat_wise = feat_wise
        if(ref_win is None and Rt is None):
            print("PLEASE PROVIDE REF WIN OR BASE RETRAINER")
        elif(ref_win is not None):
            self.n_feat = len(ref_win) if(feat_wise) else 1
            if(self.feat_wise):
                self.consec_feat_detections = [0 for _ in range(self.n_feat)]
            else:
                self.consec_feat_detections = [0]
            self.win = ref_win
            self.win_size = len(ref_win[0])
        elif(Rt is not None):
            self.n_feat = Rt.n_features if(feat_wise) else 1
            self.win_size = win_size
            if(self.feat_wise):
                self.consec_feat_detections = [0 for x in range(Rt.n_features)]
            else:
                self.consec_feat_detections = [0]
            ref_win = [list(Rt.df.drop(columns=['class']).iloc[Rt.n_train:Rt.n_train+min(win_size, Rt.n_test), feat].values)
                       for feat in range(Rt.n_features)]
            self.win = [list(Rt.df.drop(columns=['class']).iloc[Rt.n_train:Rt.n_train+min(win_size, Rt.n_test), feat].values)
                        for feat in range(Rt.n_features)]
        self.detector = alibi_detector(x_ref=np.array(ref_win).T,
                                       p_val=p_val_threshold, )
        if(len(self.win[0]) == self.win_size):
            self.window_full = True
        else:
            self.window_full = False
        self.n_consec_d = n_consec_d
        self.n_consec_w = n_consec_w
        self.clock = clock
        self.counter = 0
        self.p_val = None

    def _reset(self):
        # This is a semi-hard-_reset, nothing is kept in win
        # other alternatives can be explored.
        self.drift_detected = False
        self.warning_detected = False
        if(self.feat_wise):
            self.consec_feat_detections = [0 for _ in range(self.n_feat)]
        else:
            self.consec_feat_detections = [0]
        self.detector.x_ref = np.array(self.win).T
        self.win = [[self.win[f][-1]] for f in range(len(self.win))]
        self.window_full = False  # [Rt.X_unseen.iloc[:win_size, feat]
        # for feat in range(Rt.n_features)]  # TODO change ref_window based on last_values

    def update(self, x):  # numbers.Number) -> "DriftDetector":#TODO update with correct types
        self.counter += 1
        for feat, x_f in enumerate(x):
            self.win[feat].append(x_f)
            if(self.window_full):
                del self.win[feat][0]
        # if(np.array(self.win).ndim ==2):#TODO: Check if 1 sampled windows are not a problem
        if(self.counter % self.clock == 0):
            pred = self.detector.predict(
                np.array(self.win).T, return_p_val=True, return_distance=False)['data']
            # TODO: Check why threshold is different from p_val_threshold
            drift_detected, p_val, threshold = [
                pred[x] for x in ["is_drift", "p_val", "threshold"]]
            if(not self.feat_wise):
                p_val = [p_val]
            for feat in range(self.n_feat):
                if(p_val[feat] < self.p_val_threshold):
                    self.p_val = p_val[feat]
                    self.consec_feat_detections[feat] += 1
                    if(self.consec_feat_detections[feat] > self.n_consec_w):
                        self.warning_detected = True
                        if(self.consec_feat_detections[feat] > self.n_consec_d):
                            self.drift_detected = True
                else:
                    self.consec_feat_detections[feat] = 0

        if(not self.window_full):
            # only active while window not full
            if(len(self.win[feat]) == self.win_size):
                self.window_full = True

    def __repr__(self):
        return(f"TTest Drift detector, current window has {len(self.win[0])} elts, and the detector has seen {self.counter} points {self.p_val }")

    def drift_detected(self):
        return self.drift_detected

    def warning_detected(self):
        return self._warning_detected
