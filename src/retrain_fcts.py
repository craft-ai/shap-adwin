import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import log_loss

from lightgbm import LGBMClassifier
from math import sqrt
# import river
from river.drift import ADWIN, KSWIN, PageHinkley
from river.drift.binary import DDM, EDDM, HDDM_A


# from alibi_detect.cd import LSDDDrift, MMDDrift, ContextMMDDrift, ChiSquareDrift, KSDrift, CVMDrift
from alibi_detect.cd import KSDrift

import shap
from .detectors_fcts import *
from .shap_fcts import init_shap, compute_shap_val

import matplotlib.pyplot as plt

# import ruptures as rpt

# TODO: add option to stop retraining upon first detection


class Retrainer():
    def __init__(self, D_G):

        self.D_G = D_G
        if (not D_G.trained):
            self.D_G.train()

        # TODO: is it a good idea to double those D_G attributes ?
        self.n = D_G.n
        self.df = D_G.df
        self.n_features = D_G.n_features
        self.n_train = D_G.n_train
        self.n_test = D_G.n_test
        self.model = D_G.model
        self.trained = D_G.trained
        self.drift_points = D_G.drift_points
        self.drift_name = D_G.drift_name
        self.classif = D_G.classif
        self.objective_col = D_G.objective_col
        self.X_unseen, self.y_unseen = self.D_G.get_x_y(
            ind_start=self.n_train+self.n_test)

        self.retrain_name = "nothing"
        # if(self.classif):
        self.compute_errors()
        # else:
        # s    self.compute_preds()

        self.detection_indices = None

    def compute_errors(self):

        errors, retrain_indices = self.D_G.get_error()
        self.errors = self.D_G.errors
        return (errors, retrain_indices)

    def compute_preds(self):
        preds, retrain_indices = self.D_G.get_preds()
        self.preds = self.D_G.preds
        return (preds, retrain_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_periodic(self, period=1000, width=1000):
        model = self.model
        if (width > period):  # maybe later add support for training set larger than re-training period
            width = period

        n_unseen = self.n_train+self.n_test
        # +D_G.n_train+D_G.n_test
        retrain_indices = [n_unseen + i for i in range(period, self.n, period)]

        # initialize error
        # keep first part of errors
        errors = self.errors[:self.n_train+self.n_test]
        errors += [int(pred == label) for pred, label in zip(model.predict(self.X_unseen.iloc[0:period]),
                                                             self.y_unseen.iloc[0:period])]

        for i in range(period, len(self.X_unseen), period):
            # retrain
            # TODO: put model class init function here to generalize to other models
            model = LGBMClassifier(random_state=self.D_G.model_random_seed, verbose=-1)
            model.fit(self.X_unseen.iloc[max(0, i-width): i],
                      self.y_unseen.iloc[max(0, i-width): i])

            # compute errors
            preds = model.predict(
                self.X_unseen.iloc[max(0, i):i+period].values)
            errors += [int(p == y) for p, y in zip(preds,
                                                   self.y_unseen.iloc[max(0, i):min(len(self.y_unseen), i+period)])]

        self.errors = errors
        self.detection_indices = retrain_indices
        self.retrain_name = "periodic"
        return (errors, retrain_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------
    # TODO: Same as stack train

    def retrain_periodic_increase(self, period=1000, width=500):
        model = self.model
        if (width > period):  # maybe later add support for training set larger than re-training period
            width = period

        X_monitor, y_unseen = self.D_G.get_x_y(
            ind_start=self.n_train+self.n_test)
        # initialize train set
        X_train, y_train = self.D_G.get_x_y(ind_end=self.n_train)

        # +D_G.n_train+D_G.n_test
        retrain_indices = [i for i in range(period, self.n, period)]

        # initialize error
        preds = model.predict(X_monitor.iloc[0:period].values)
        errors = [int(p == y) for p, y in zip(preds, y_unseen.iloc[0:period])]

        for i in range(period, len(X_monitor), period):
            # retrain
            model = LGBMClassifier(random_state=self.D_G.model_random_seed, verbose=-1)

            X_train = pd.concat([X_train, X_monitor.iloc[max(0, i-width): i]])
            y_train = pd.concat([y_train, y_unseen.iloc[max(0, i-width): i]])
            model.fit(X_train, y_train)

            # compute errors
            preds = model.predict(X_monitor.iloc[max(0, i):i+period].values)
            errors += [int(p == y) for p, y in zip(preds,
                                                   y_unseen.iloc[max(0, i):min(len(y_unseen), i+period)])]

        self.errors = errors
        return (errors, retrain_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_feat_drift(self, win_size=100, p_val_threshold=1e-10):
        '''
            retrain new model when a drift is detected in the one of the feature distributions.

            win_size: size of the windows to compute student test on.

        '''

        if (not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        errors = self.errors[:self.n_train+self.n_test]
        detection_indices = []
        # drift detection
        # number of consecutive detection
        consec = [0 for x in range(self.n_features)]

        ref_win = [self.X_unseen.iloc[:win_size, feat]
                   for feat in range(self.n_features)]

        # initialize error
        errors = [int(pred == label) for pred, label in zip(model.predict(
            self.X_unseen.iloc[0:win_size]), self.y_unseen.iloc[0:win_size])]

        for index in range(win_size, len(self.X_unseen)):
            # get next sample
            x, label = self.X_unseen.iloc[index], self.y_unseen.iloc[index]
            errors.append(int(model.predict([x]) == label))

        for index in range(win_size, len(self.X_unseen)):
            # get next sample
            x, label = self.X_unseen.iloc[index], self.y_unseen.iloc[index]
            errors.append(int(model.predict([x]) == label))
            win = self.X_unseen.iloc[index-win_size:index]
            for feat in range(self.n_features):
                p_val = ttest_ind(win[feat], ref_win[feat])[1]
                if (p_val < p_val_threshold):
                    consec[feat] += 1
                else:
                    consec[feat] = 0

                if (consec[feat] > win_size):
                    n_samples_retrain = min(index, win_size)

                    X_train = self.X_unseen.iloc[index -
                                                 n_samples_retrain:index]
                    y_train = self.y_unseen.iloc[index -
                                                 n_samples_retrain:index]

                    model = LGBMClassifier(
                        random_state=self.D_G.model_random_seed, verbose=-1)
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    consec = [0 for x in range(self.n_features)]
                    ref_win = [X_train.iloc[:win_size, feat]
                               for feat in range(self.n_features)]

        self.errors = errors
        return (errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------
    # TODO: Same as stack train

    def retrain_feat_model_increase(self, win_size=100, p_val_threshold=1e-10, n_estimators=50, n_esti_inc=5):
        # train initial model

        model = RandomForestClassifier(
            n_estimators=n_estimators)  # , warm_start=True)
        X_train, y_train = self.D_G.get_x_y(ind_end=self.n_train)
        # X_test, y_test = self.D_G.get_x_y(ind_strat=self.n_train, ind_end=self.n_train+self.n_test)
        model.fit(X_train, y_train)

        # prepare returns
        errors = self.errors[:self.n_train+self.n_test]
        detection_indices = []
        # drift detection
        # number of consecutive detection
        consec = [0 for x in range(self.n_features)]
        # ind_detect = [0 for feat in range(self.n_features)]#can be used as warning index
        ref_win = [self.X_unseen.iloc[:win_size, feat]
                   for feat in range(self.n_features)]

        # initialize error
        errors = [int(pred == label) for pred, label in zip(model.predict(
            self.X_unseen.iloc[0:win_size]), self.y_unseen.iloc[0:win_size])]

        for index in range(win_size, len(self.X_unseen)):
            # get next sample
            x, label = self.X_unseen.iloc[index], self.y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)

            for feat in range(self.n_features):
                win = self.X_unseen.iloc[index-win_size:index, feat]
                p_val = ttest_ind(win, ref_win[feat])[1]
                if (p_val < p_val_threshold):
                    consec[feat] += 1
                    # if(ind_detect[feat] == 0):
                    #    ind_detect[feat] = index
                else:
                    consec[feat] = 0
                    # ind_detect[feat] = 0

                if (consec[feat] > win_size):  # train a single tree and add it to the ensemble
                    n_samples_retrain = min(index, win_size)

                    X_train = pd.concat(
                        [X_train, self.X_unseen.iloc[index - n_samples_retrain:index]])
                    y_train = pd.concat(
                        [y_train, self.y_unseen.iloc[index - n_samples_retrain:index]])
                    # TODO: add function init_model()
                    model = RandomForestClassifier(n_estimators=n_estimators)
                    # growing_rf.n_estimators += n_esti_inc
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    consec = [0 for x in range(self.n_features)]
                    ref_win = [X_train.iloc[:win_size, feat]
                               for feat in range(self.n_features)]

        self.errors = errors
        return (errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_basic_shap(self, win_size=500):
        """
        TODO: Check what this is doing and re-implement it
        -> TTest on each feat shap vals

        """

        model = self.model

        # init shap explainer and reference window
        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(columns=['class'])
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(self.X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            self.X_unseen[:win_size], self.y_unseen[:win_size])

        errors = self.errors[:self.n_train+self.n_test]
        warning_indices = []
        detection_indices = []

        # number of consecutive detection
        consec = [0 for x in range(self.n_features)]
        ind_detect = [0 for feat in range(self.n_features)]

        ref_win = [shap_values[:win_size, feat]
                   for feat in range(self.n_features)]
        thresh = [np.quantile(ref, 0.75) for ref in ref_win]
        ref_win = [list(filter(lambda x: x > thresh[i], ref))
                   for i, ref in enumerate(ref_win)]

        # initialize error
        errors = [int(pred == label) for pred, label in zip(model.predict(
            self.X_unseen.iloc[0:win_size]), self.y_unseen.iloc[0:win_size])]

        for index in range(win_size, len(self.X_unseen)):
            # get next sample
            x, label = self.X_unseen.iloc[index], self.y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                self.X_unseen.iloc[[index]], self.y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                win = shap_values[index - win_size:index, feat]
                thresh = np.quantile(win, 0.75)
                win = list(filter(lambda x: x > thresh, win))
                p_val = ttest_ind(win, ref_win[feat])[1]
                if (p_val < 1e-5):
                    consec[feat] += 1
                    if (ind_detect[feat] == 0):
                        ind_detect[feat] = index
                else:
                    consec[feat] = 0
                    ind_detect[feat] = 0
                if (consec[feat] > 10):  # retrain
                    n_samples_retrain = min(index, win_size)
                    model = LGBMClassifier(
                        random_state=self.D_G.model_random_seed, verbose=-1)
                    X_train = self.X_unseen.iloc[index -
                                                 n_samples_retrain:index]
                    y_train = self.y_unseen.iloc[index -
                                                 n_samples_retrain:index]
                    model.fit(X_train, y_train)

                    # reset detection
                    consec = [0 for x in range(self.n_features)]
                    detection_indices.append(index)
                    # recompute Shapley values
                    explainerError = shap.TreeExplainer(model, data=self.X_unseen.iloc[index - n_samples_retrain:index],
                                                        feature_perturbation="interventional", model_output='log_loss')
                    shap_values[index:index+win_size, :] = explainerError.shap_values(
                        self.X_unseen[index:index+win_size], self.y_unseen[index:index+win_size])
                    ref_win = [shap_values[index: index+win_size,  feat]
                               for feat in range(self.n_features)]
                    thresh = [np.quantile(ref, 0.75) for ref in ref_win]
                    ref_win = [list(filter(lambda x: x > thresh[i], ref))
                               for i, ref in enumerate(ref_win)]

        self.errors = errors
        return (errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_adwin(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=ADWIN,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": beta_d, "clock": 1},
                                           warning_params={
                                               "delta": beta_w, "clock": 1},
                                           win_size=win_size,
                                           retrain_name="adwin_loss_shap",
                                           bgd_type="train",
                                           stop_first_detect=stop_first_detect))
# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_adwin_back_worse(self,  delta_w=0.01, delta_d=0.002, win_size=200, clock=1, return_shap=False, stop_first_detect=False):
        """
        shap explainer background data is composed of 50 points with the highest loss

        """
        return (self.retrain_detector_shap(detector_func=ADWIN,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": delta_d, "clock": clock},
                                           warning_params={
                                               "delta": delta_w, "clock": clock},
                                           win_size=win_size,
                                           retrain_name="adwin_loss_shap",
                                           bgd_type="worse",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_adwin_back_best(self, delta_w=0.01, delta_d=0.002, clock=1, win_size=200, return_shap=False, stop_first_detect=False):
        """
            background is filled with with the 50 lowest loss points
        """
        return (self.retrain_detector_shap(detector_func=ADWIN,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": delta_d, "clock": clock},
                                           warning_params={
                                               "delta": delta_w, "clock": clock},
                                           win_size=win_size,
                                           retrain_name="adwin_loss_shap",
                                           bgd_type="best",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_adwin_experimental(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False):

        # init shap explainer and reference window
        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(
            columns=['class']).sample(min(50, self.n_train+self.n_test))
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(self.X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            self.X_unseen[:win_size], self.y_unseen[:win_size])
        shap_init = shap_values[:win_size]

        errors = self.errors[:self.n_train+self.n_test]
        detection_indices = []
        # detection init
        warnings_feat = [ADWIN(beta_w) for feat in range(self.n_features)]
        detect_feat = [ADWIN(beta_d) for feat in range(self.n_features)]
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(model.predict(
            self.X_unseen.iloc[0:win_size]), self.y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])

        for index in range(win_size, len(self.X_unseen)):
            # get next sample
            x, label = self.X_unseen.iloc[index], self.y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                self.X_unseen.iloc[[index]], self.y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])
                # print(shap_values[index,feat],feat)
                if warnings_feat[feat].detected_change():  # warning detection
                    if (first_warning > index):
                        first_warning = index
                if detect_feat[feat].detected_change():  # drift detection
                    n_samples_retrain = min(index, max(
                        index-first_warning, win_size))
                    # here we must fork wether we increase or retrain.

                    shap_before_warn = shap_values[first_warning -
                                                   min(win_size, index):first_warning]
                    shap_problem = shap_values[index - n_samples_retrain:index]
                    # print("feature %d"%feat)

                    shaps = [x[:, feat]
                             for x in [shap_init, shap_before_warn, shap_problem]]
                    # print([np.shape(x) for x in shaps])
                    means = [(i, np.mean(x)) for i, x in enumerate(shaps)]
                    quantiles = [(i, np.quantile(x, 0.9))
                                 for i, x in enumerate(shaps)]
                    means = sorted(means, key=lambda x: x[1])
                    quantiles = sorted(quantiles, key=lambda x: x[1])
                    # print(means,quantiles)
                    print([x[0] for x in means].index(2), [x[0]
                          for x in quantiles].index(2))
                    if (([x[0] for x in means].index(2)+[x[0] for x in quantiles].index(2)) < 1):
                        print("THIS IS GETTING BETTER DO NOT INTERVENE !!!!",
                              index, n_samples_retrain)
                        for feat in range(self.n_features):
                            warnings_feat[feat].reset()
                            detect_feat[feat].reset()
                        first_warning = np.inf
                    else:
                        X_train = self.X_unseen.iloc[index -
                                                     n_samples_retrain:index]
                        y_train = self.y_unseen.iloc[index -
                                                     n_samples_retrain:index]

                        model = LGBMClassifier(
                            random_state=self.D_G.model_random_seed, verbose=-1)
                        model.fit(X_train, y_train)

                        detection_indices.append(index)
                        for feat in range(self.n_features):
                            warnings_feat[feat].reset()
                            detect_feat[feat].reset()
                        first_warning = np.inf
                    break  # skip other features as we re-trained our model.

        self.errors = errors
        if (return_shap):
            return (errors, detection_indices, shap_values)
        else:
            return (errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_adwin_back_train(self, delta_w=0.01, delta_d=0.002, clock=1, win_size=200, return_shap=False, stop_first_detect=False):
        """
        TODO: check of what to do with duplicate func retrain_shap_adwin
        """

        return (self.retrain_detector_shap(detector_func=ADWIN,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": delta_d, "clock": clock},
                                           warning_params={
                                               "delta": delta_w, "clock": clock},
                                           win_size=win_size,
                                           retrain_name="adwin_loss_shap",
                                           bgd_type="train",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_adwin_smallback(self, beta_w=0.01, beta_d=0.002, clock=1, win_size=200, return_shap=False, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=ADWIN,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": beta_d, "clock": clock},
                                           warning_params={
                                               "delta": beta_w, "clock": clock},
                                           win_size=win_size,
                                           retrain_name="adwin_loss_shap",
                                           bgd_type="small",
                                           stop_first_detect=stop_first_detect))

# ____________________--

    def retrain_shap_PH(self, delta_w=0.01, delta_d=0.005, win_size=200, clock=1, bgd_type="sample",
                        n_samp_bgd=25, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={"delta": delta_d},
                                           warning_params={"delta": delta_w},
                                           win_size=win_size,
                                           retrain_name="PH_logloss_shap",
                                           bgd_type="sample",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))
# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_PH_back_worse(self,  delta_w=0.01, delta_d=0.002, win_size=200, return_shap=False, stop_first_detect=False):
        """
        shap explainer background data is composed of 50 points with the highest loss
        """
        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": delta_d},
                                           warning_params={
                                               "delta": delta_w},
                                           win_size=win_size,
                                           retrain_name="PH_loss_shap",
                                           bgd_type="worse",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_PH_back_best(self, delta_w=0.01, delta_d=0.002, win_size=200, return_shap=False, stop_first_detect=False):
        """
            background is filled with with the 50 lowest loss points
        """
        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": delta_d},
                                           warning_params={
                                               "delta": delta_w},
                                           win_size=win_size,
                                           retrain_name="PH_loss_shap",
                                           bgd_type="best",
                                           n_samp_bgd=25,  # TODO: pass n_bkg samp as param
                                           stop_first_detect=stop_first_detect))

    def retrain_shap_PH_back_train(self, delta_w=0.01, delta_d=0.002, win_size=200, return_shap=False, stop_first_detect=False):
        """
        TODO: check of what to do with duplicate func retrain_shap_PH
        """

        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": delta_d},
                                           warning_params={
                                               "delta": delta_w},
                                           win_size=win_size,
                                           retrain_name="PH_loss_shap",
                                           bgd_type="train",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_PH_smallback(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": beta_d},
                                           warning_params={
                                               "delta": beta_w},
                                           win_size=win_size,
                                           retrain_name="PH_loss_shap",
                                           bgd_type="small",
                                           stop_first_detect=stop_first_detect))

    def retrain_shap_KSWIN(self, alpha=0.005, w_size=100, stat_size=30, win_size=200, return_shap=False, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=KSWIN,
                                           signal="log_loss",
                                           detector_params={"alpha": alpha,
                                                            "window_size": w_size,
                                                            "stat_size": stat_size},
                                           warning_params={},
                                           win_size=win_size,
                                           retrain_name="KSWIN_loss_shap",
                                           bgd_type="train",
                                           stop_first_detect=stop_first_detect))
# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_KSWIN_back_worse(self, alpha=0.005, w_size=100, stat_size=30, win_size=200, return_shap=False, stop_first_detect=False):
        """
        shap explainer background data is composed of 50 points with the highest loss

        """
        return (self.retrain_detector_shap(detector_func=KSWIN,
                                           signal="log_loss",
                                           detector_params={"alpha": alpha,
                                                            "window_size": w_size,
                                                            "stat_size": stat_size},
                                           warning_params={},
                                           win_size=win_size,
                                           retrain_name="KSWIN_loss_shap",
                                           bgd_type="worse",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_KSWIN_back_best(self, alpha=0.005, w_size=100, stat_size=30, win_size=200, return_shap=False, stop_first_detect=False):
        """
            background is filled with with the 50 lowest loss points
        """
        return (self.retrain_detector_shap(detector_func=KSWIN,
                                           signal="log_loss",
                                           detector_params={"alpha": alpha,
                                                            "window_size": w_size,
                                                            "stat_size": stat_size},
                                           warning_params={},
                                           win_size=win_size,
                                           retrain_name="KSWIN_loss_shap",
                                           bgd_type="best",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))

    def retrain_shap_KSWIN_back_train(self, alpha=0.005, w_size=100, stat_size=30, win_size=200, return_shap=False, stop_first_detect=False):
        """
        TODO: check of what to do with duplicate func retrain_shap_KSWIN
        """

        return (self.retrain_detector_shap(detector_func=KSWIN,
                                           signal="log_loss",
                                           detector_params={"alpha": alpha,
                                                            "window_size": w_size,
                                                            "stat_size": stat_size},
                                           warning_params={},
                                           win_size=win_size,
                                           retrain_name="KSWIN_loss_shap",
                                           bgd_type="train",
                                           stop_first_detect=stop_first_detect))


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_KSWIN_smallback(self, alpha=0.005, w_size=100, stat_size=30, win_size=200, return_shap=False, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=KSWIN,
                                           signal="log_loss",
                                           detector_params={"alpha": alpha,
                                                            "window_size": w_size,
                                                            "stat_size": stat_size},
                                           warning_params={},
                                           win_size=win_size,
                                           retrain_name="KSWIN_loss_shap",
                                           bgd_type="small",
                                           stop_first_detect=stop_first_detect))
# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_ks(self, delta_w=0.1, delta_d=0.01, win_size=200, clock=1):
        return (self.retrain_alibi_detector(p_val_threshold=1e-9,
                                            alibi_detector=KSDrift,
                                            win_size=win_size,
                                            n_consec_w=10,
                                            n_consec_d=3))

    def retrain_adwin(self, delta_w=0.1, delta_d=0.01, win_size=200, clock=1, stop_first_detect=False):

        return (self.retrain_detector(detector_func=ADWIN,
                                      signal="error",
                                      detector_params={
                                          "delta": delta_d, "clock": clock},
                                      warning_params={
                                          "delta": delta_w, "clock": clock},
                                      win_size=win_size,
                                      retrain_name="adwin",
                                      stop_first_detect=stop_first_detect))

    # TODO: def retrain_PH parameters
    def retrain_PH(self, delta_w=0.01, delta_d=0.005, win_size=200, min_instances=30, threshold=50.0, alpha=0.9999, stop_first_detect=False):
        return (self.retrain_detector(detector_func=PageHinkley,
                                      signal="error",
                                      detector_params={
                                          "delta": delta_d,
                                          "threshold": threshold,
                                          "alpha": alpha,
                                          "min_instances": min_instances},
                                      warning_params={
                                          "delta": delta_w, },
                                      win_size=win_size,
                                      retrain_name="PH",
                                      stop_first_detect=stop_first_detect))
    # TODO: def retrain_KSWIN parameters

    def retrain_KSWIN(self, win_size=200, alpha=0.005, w_size=100, stat_size=30, stop_first_detect=False):
        return (self.retrain_detector(detector_func=KSWIN,
                                      signal="error",
                                      detector_params={"alpha": alpha,
                                                       "window_size": w_size,
                                                       "stat_size": stat_size},
                                      warning_params={},
                                      win_size=win_size,
                                      retrain_name="KSWIN",
                                      stop_first_detect=stop_first_detect))
    # TODO: def retrain_DDM parameters

    def retrain_DDM(self, win_size=200, stop_first_detect=False):
        """TODO: Check cold start param"""
        return (self.retrain_detector(detector_func=DDM,
                                      signal="error",
                                      detector_params={},
                                      warning_params={},
                                      win_size=win_size,
                                      retrain_name="DDM",
                                      stop_first_detect=stop_first_detect))
    # TODO: def retrain_EDDM parameters

    def retrain_EDDM(self, win_size=200, stop_first_detect=False):
        return (self.retrain_detector(detector_func=EDDM,
                                      signal="error",
                                      detector_params={},
                                      warning_params={},
                                      win_size=win_size,
                                      retrain_name="EDDM",
                                      stop_first_detect=stop_first_detect))
    # TODO: def retrain_HDDM_A parameters

    def retrain_HDDM_A(self, win_size=200, stop_first_detect=False):
        return (self.retrain_detector(detector_func=HDDM_A,
                                      signal="error",
                                      detector_params={},
                                      warning_params={},
                                      win_size=win_size,
                                      retrain_name="HDDM_A",
                                      stop_first_detect=stop_first_detect))

    def retrain_ttest(self, p_val_threshold=1e-4, clock=1, win_size=200, stop_first_detect=False):
        ref_win = [list(self.df.drop(columns=['class']).iloc[self.n_train:self.n_train+min(win_size, self.n_test),
                                                             feat].values)
                   for feat in range(self.n_features)]
        return (self.retrain_detector(detector_func=Ttest_detector,
                                      signal='point',
                                      detector_params={"ref_win": ref_win,
                                                       "Rt": self,
                                                       "win_size": win_size,
                                                       "clock": clock,
                                                       "p_val_threshold": p_val_threshold},
                                      warning_inside=True,
                                      retrain_name="ttest",
                                      stop_first_detect=stop_first_detect))

    def retrain_alibi_detector(self, p_val_threshold=1e-4, clock=1, win_size=200,
                               alibi_detector=KSDrift,
                               n_consec_w=1, n_consec_d=3):

        ref_win = [list(self.df.drop(columns=['class']).iloc[self.n_train:self.n_train+min(win_size, self.n_test),
                                                             feat].values)
                   for feat in range(self.n_features)]
        if (alibi_detector.__name__ in ["LSDDDrift", "MMDDrift", "ContextMMDDrift"]):
            feat_wise = False
        elif (alibi_detector.__name__ in ["ChiSquareDrift", "KSDrift", "CVMDrift"]):
            feat_wise = True
        else:
            print("THIS ALIBI DETECTOR is not accepted YET")
        return (self.retrain_detector(detector_func=Alibi_generic_detector,
                                      signal='point',
                                      detector_params={"alibi_detector": alibi_detector,  # Alibi_generic_detector#KSDrift
                                                       "feat_wise": feat_wise,
                                                       "ref_win": ref_win,
                                                       "win_size": win_size,
                                                       "n_consec_w": n_consec_w,
                                                       "n_consec_d": n_consec_d,
                                                       "clock": clock,
                                                       "p_val_threshold": p_val_threshold},
                                      warning_inside=True,
                                      retrain_name=alibi_detector.__name__))

#
# from alibi_detect.cd import LSDDDrift, MMDDrift, ContextMMDDrift
# #LearnedKernelDrift -> requires kernel argument
# #ContextMMDDrift -> requires c_ref argument
# #SpotTheDiffDrift -> error strange
# from alibi_detect.cd import ChiSquareDrift, KSDrift, CVMDrift, FETDrift
# #FETDrift -> ValueError: `alternative` must be either 'greater', 'less' or 'two-sided'


# ------------------------------------------------------------------------------------------------------------------------------------------------
    # TODO: do stack train model retraining heuristic in function

    def retrain_adwin_stack_train(self, delta_w=0.01, delta_d=0.002, win_size=200):

        X_train, y_train = self.D_G.get_x_y(ind_end=self.n_train)

        errors = self.errors[:self.n_train+self.n_test]
        detection_indices = []
        # detection init
        warning = ADWIN(delta_w)
        detect = ADWIN(delta_d)
        first_warning = np.inf

        for index in range(0, len(self.X_unseen)):
            # get next sample
            x, label = self.X_unseen.iloc[index], self.y_unseen.iloc[index]
            proba = model.predict_proba([x])[0][1 if label else 0]
            loss = -(label*np.log(proba))+(1 - label)*np.log(1 - proba)
            error = int(model.predict([x]) == label)
            errors.append(error)

            warning.add_element(loss)
            detect.add_element(loss)
            # print(shap_values[index,feat],feat)
            if warning.detected_change():  # warning detection
                if (first_warning > index):
                    first_warning = index
            if detect.detected_change():  # warning detection
                n_samples_retrain = min(index, max(
                    index-first_warning, win_size))

                X_train = pd.concat(
                    [X_train, self.X_unseen.iloc[index - n_samples_retrain:index]])
                y_train = pd.concat(
                    [y_train, self.y_unseen.iloc[index - n_samples_retrain:index]])

                model = LGBMClassifier(random_state=self.D_G.model_random_seed, verbose=-1)
                model.fit(X_train, y_train)

                detection_indices.append(index)
                for feat in range(self.n_features):
                    warning.reset()
                    detect.reset()
                first_warning = np.inf
                break

        self.errors = errors
        # ------------------------------------------------------------------------------------------------------------------------------------------------
        return (errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_adwin_stack(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False):

        X_train, y_train = self.D_G.get_x_y(ind_end=self.n_train)
        # init shap explainer and reference window
        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(
            columns=['class']).sample(min(50, self.n_train+self.n_test))
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(self.X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            self.X_unseen[:win_size], self.y_unseen[:win_size])

        errors = self.errors[:self.n_train+self.n_test]
        detection_indices = []
        # detection init
        warnings_feat = [ADWIN(beta_w) for feat in range(self.n_features)]
        detect_feat = [ADWIN(beta_d) for feat in range(self.n_features)]
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(model.predict(
            self.X_unseen.iloc[0:win_size]), self.y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])

        for index in range(win_size, len(self.X_unseen)):
            # get next sample
            x, label = self.X_unseen.iloc[index], self.y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                self.X_unseen.iloc[[index]], self.y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])
                # print(shap_values[index,feat],feat)
                if warnings_feat[feat].detected_change():  # warning detection
                    if (first_warning > index):
                        first_warning = index
                if detect_feat[feat].detected_change():  # warning detection
                    n_samples_retrain = min(index, max(
                        index-first_warning, win_size))

                    X_train = pd.concat(
                        [X_train, self.X_unseen.iloc[index - n_samples_retrain:index]])
                    y_train = pd.concat(
                        [y_train, self.y_unseen.iloc[index - n_samples_retrain:index]])

                    model = LGBMClassifier(
                        random_state=self.D_G.model_random_seed, verbose=-1)
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    for feat in range(self.n_features):
                        warnings_feat[feat].reset()
                        detect_feat[feat].reset()
                    first_warning = np.inf
                    break

        self.errors = errors
        if (return_shap):
            return (errors, detection_indices, shap_values)
        else:
            return (errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------


#    def retrain_shap_KSWIN(self, alpha_w=0.01, alpha_d=0.002, win_size=200, return_shap=False):
#
#        # init shap explainer and reference window
#        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(columns=['class'])
#        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
#                                            feature_perturbation="interventional", model_output='log_loss')
#        shap_values = np.zeros((len(self.X_unseen), self.n_features))
#        shap_values[:win_size] = explainerError.shap_values(
#            self.X_unseen[:win_size], self.y_unseen[:win_size])
#
#        errors = self.errors[:self.n_train+self.n_test]
#        detection_indices = []
#        # detection init
#        warnings_feat = [KSWIN(alpha_w) for feat in range(self.n_features)]
#        detect_feat = [KSWIN(alpha_d) for feat in range(self.n_features)]
#        first_warning = np.inf
#
#        # initialize error
#        errors = [int(pred == label) for pred, label in zip(model.predict(
#            self.X_unseen.iloc[0:win_size]), self.y_unseen.iloc[0:win_size])]
#
#        # initialize detectors
#        for index in range(win_size):
#            for feat in range(self.n_features):
#                warnings_feat[feat].add_element(shap_values[index, feat])
#                detect_feat[feat].add_element(shap_values[index, feat])
#
#        for index in range(win_size, len(self.X_unseen)):
#            # get next sample
#            x, label = self.X_unseen.iloc[index], self.y_unseen.iloc[index]
#            error = int(model.predict([x]) == label)
#            errors.append(error)
#            shap_values[index] = explainerError.shap_values(
#                self.X_unseen.iloc[[index]], self.y_unseen.iloc[[index]])[0]
#
#            for feat in range(self.n_features):
#                warnings_feat[feat].add_element(shap_values[index, feat])
#                detect_feat[feat].add_element(shap_values[index, feat])
#                # print(shap_values[index,feat],feat)
#                if warnings_feat[feat].detected_change():  # warning detection
#                    if (first_warning > index):
#                        first_warning = index
#                if detect_feat[feat].detected_change():  # warning detection
#                    n_samples_retrain = min(index, max(
#                        index-first_warning, win_size))
#
#                    X_train = self.X_unseen.iloc[index -
#                                                 n_samples_retrain:index]
#                    y_train = self.y_unseen.iloc[index -
#                                                 n_samples_retrain:index]
#
#                    model = LGBMClassifier(
#                        random_state=self.D_G.model_random_seed, verbose=-1)
#                    model.fit(X_train, y_train)
#
#                    detection_indices.append(index)
#                    for feat in range(self.n_features):
#                        warnings_feat[feat].reset()
#                        detect_feat[feat].reset()
#                    first_warning = np.inf
#                    break
#
#        self.errors = errors
#        if (return_shap):
#            return (errors, detection_indices, shap_values)
#        else:
#            return (errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------


#    def retrain_shap_PH(self, delta_w=0.01, delta_d=0.005, win_size=200, return_shap=False):
#
#        # init shap explainer and reference window
#        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(columns=['class'])
#        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
#                                            feature_perturbation="interventional", model_output='log_loss')
#        shap_values = np.zeros((len(self.X_unseen), self.n_features))
#        shap_values[:win_size] = explainerError.shap_values(
#            self.X_unseen[:win_size], self.y_unseen[:win_size])
#
#        errors = self.errors[:self.n_train+self.n_test]
#        detection_indices = []
#        # detection init
#        warnings_feat = [PageHinkley(delta_w)
#                         for feat in range(self.n_features)]
#        detect_feat = [PageHinkley(delta_d) for feat in range(self.n_features)]
#        first_warning = np.inf
#
#        # initialize error
#        errors = [int(pred == label) for pred, label in zip(model.predict(
#            self.X_unseen.iloc[0:win_size]), self.y_unseen.iloc[0:win_size])]
#
#        # initialize detectors
#        for index in range(win_size):
#            for feat in range(self.n_features):
#                warnings_feat[feat].add_element(shap_values[index, feat])
#                detect_feat[feat].add_element(shap_values[index, feat])
#
#        for index in range(win_size, len(self.X_unseen)):
#            # get next sample
#            x, label = self.X_unseen.iloc[index], self.y_unseen.iloc[index]
#            error = int(model.predict([x]) == label)
#            errors.append(error)
#            shap_values[index] = explainerError.shap_values(
#                self.X_unseen.iloc[[index]], self.y_unseen.iloc[[index]])[0]
#
#            for feat in range(self.n_features):
#                warnings_feat[feat].add_element(shap_values[index, feat])
#                detect_feat[feat].add_element(shap_values[index, feat])
#                # print(shap_values[index,feat],feat)
#                if warnings_feat[feat].detected_change():  # warning detection
#                    if (first_warning > index):
#                        first_warning = index
#                if detect_feat[feat].detected_change():  # warning detection
#                    n_samples_retrain = min(index, max(
#                        index-first_warning, win_size))
#
#                    X_train = self.X_unseen.iloc[index -
#                                                 n_samples_retrain:index]
#                    y_train = self.y_unseen.iloc[index -
#                                                 n_samples_retrain:index]
#
#                    model = LGBMClassifier(
#                        random_state=self.D_G.model_random_seed)
#                    model.fit(X_train, y_train)
#
#                    detection_indices.append(index)
#                    for feat in range(self.n_features):
#                        warnings_feat[feat].reset()
#                        detect_feat[feat].reset()
#                    first_warning = np.inf
#                    break
#
#        self.errors = errors
#        if (return_shap):
#            return (errors, detection_indices, shap_values)
#        else:
#            return (errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    # TODO: Consider initializing detector on train and/or test set
    # TODO: Examine Why the retrain funcs are not deterministic.

    def retrain_detector(self, detector_func, detector_params=None, warning_params=None, signal="error",
                         warning_inside=False, win_size=200, retrain_name="detector", stop_first_detect=False):
        """Method working with a dectector provided as in RIVER
                riverml.xyz
          """
        if (detector_params == None):
            detector = detector_func()
        else:
            detector = detector_func(**detector_params)
        if (not warning_inside):
            if (warning_params == None):
                warning = detector_func()
            else:
                warning = detector_func(**warning_params)

        # keep first part of errors
        errors = self.errors[:self.n_train+self.n_test]
        detection_indices = []
        first_warning = np.inf

        for index, (x, label) in enumerate(zip(self.X_unseen.values, self.y_unseen.values)):
            if (self.classif):
                error = int(self.model.predict([x]) == label)
            else:
                error = abs(self.model.predict([x]) - label)
            errors.append(error)

            if (signal == "log_loss"):
                # [1 if label else 0]#TODO: Fix Log_loss
                mod_probas = self.model.predict_proba([x])
                proba = mod_probas[0][mod_probas.argmax()]
                is_good_label = 1 if label == mod_probas.argmax() else 0
                # TODO: add good_label bellow
                log_loss = -(label*np.log(proba))+(1 - label)*np.log(1 - proba)
                detector.update(log_loss)
                if (not warning_inside):
                    warning.update(log_loss)

            elif (signal == "point"):
                detector.update(x)
            elif (signal == "error"):
                detector.update(error)
                if (not warning_inside):
                    warning.update(error)
            else:  # default signal is error
                detector.update(error)
                if (not warning_inside):
                    warning.update(error)

            if warning_inside:  # warning detection
                if (detector.warning_detected and first_warning > index):
                    first_warning = index
            else:
                if warning.drift_detected and first_warning > index:
                    first_warning = index
            if detector.drift_detected:  # true detection
                true_index = self.n_train+self.n_test + index
                detection_indices.append(true_index)
                if (stop_first_detect):
                    self.errors = errors
                    self.detection_indices = detection_indices
                    self.retrain_name = retrain_name
                return (errors, detection_indices)
                # define on which batch of data we retrain.#TODO: write generic retraining function
                # if(detector_params == None):
                detector._reset()  # detector_func()
                # else:
                # detector = detector_func(**detector_params)
                if (not warning_inside):
                    warning._reset()
                    # if(warning_params == None):
                    #    warning = detector_func()
                    # else:
                    #    warning = detector_func(**warning_params)
                first_warning = np.inf

                n_sample_retrain = min(
                    max(index-first_warning, win_size), true_index)
                X_train, y_train = self.D_G.get_x_y(
                    ind_start=true_index-n_sample_retrain, ind_end=true_index)
                self.model = LGBMClassifier(
                    random_state=self.D_G.model_random_seed, verbose=-1)
                self.model.fit(X_train, y_train)

        self.errors = errors
        self.detection_indices = detection_indices
        self.retrain_name = retrain_name
        return (errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------
    # TODO: adapt this function to work with a detector on each shap feature loss.

    def retrain_detector_shap(self, detector_func, detector_params=None, warning_params=None, signal="error",
                              warning_inside=False, win_size=200, retrain_name="detector",
                              bgd_type=None, n_samp_bgd=None, stop_first_detect=False):
        """Method working with a dectector provided as in RIVER
                riverml.xyz
          """
        if (detector_params == None):
            detectors = [detector_func() for _ in range(self.n_features)]
        else:
            detectors = [detector_func(**detector_params)
                         for _ in range(self.n_features)]
        if (not warning_inside):
            if (warning_params == None):
                warnings = [detector_func() for _ in range(self.n_features)]
            else:
                warnings = [detector_func(**warning_params)
                            for _ in range(self.n_features)]

        # keep first part of errors
        errors = self.errors[:self.n_train+self.n_test]
        detection_indices = []
        first_warning = np.inf

        if (bgd_type == "best" or bgd_type == "worse"):
            X_test, y_test = self.D_G.get_x_y(ind_end=self.n_train+self.n_test)
            probas = [self.model.predict_proba([x])[0][1 if y else 0]
                      for x, y in zip(X_test.values, y_test)]
            loss = [-(label*np.log(proba))+(1-label)*np.log(1 - proba)
                    for proba, label in zip(probas, y_test)]
        else:
            loss = None
        explainer = init_shap(
            signal, self, bgd_type=bgd_type, n_samp_bgd=n_samp_bgd, loss=loss, objective_col=self.objective_col)

        for index, (x, label) in enumerate(zip(self.X_unseen.values, self.y_unseen.values)):
            if (self.classif):
                error = int(self.model.predict([x]) == label)
            else:
                error = abs(self.model.predict([x]) - label)
            errors.append(error)

            if (signal == "log_loss"):
                # requires bgrd,
                # print("not yet implemented")
                point_shap_val = compute_shap_val(
                    signal, self, explainer, x, label)
                for d, feat_shapval in zip(detectors, point_shap_val):
                    d.update(feat_shapval)
                if (not warning_inside):
                    for w, feat_shapval in zip(warnings, point_shap_val):
                        w.update(feat_shapval)
            elif (signal == "point"):  # TODO: add all_feats support for shap
                print("NOT SUPPORTED YETs")
                # detector.update(x)
            if warning_inside:  # warning detection
                if (any([d.warning_detected for d in detectors]) and first_warning > index):
                    first_warning = index
            else:
                if any([w.drift_detected for w in warnings]) and first_warning > index:
                    first_warning = index
            if any([d.drift_detected for d in detectors]):  # true detection
                true_index = self.n_train+self.n_test + index
                detection_indices.append(true_index)
                if (stop_first_detect):
                    self.errors = errors
                    self.detection_indices = detection_indices
                    self.retrain_name = retrain_name
                    return (errors, detection_indices)

                for d in detectors:
                    d._reset()
                if (not warning_inside):
                    for w in warnings:
                        w._reset()
                first_warning = np.inf

                n_sample_retrain = min(
                    max(index-first_warning, win_size), true_index)
                X_train, y_train = self.D_G.get_x_y(
                    ind_start=true_index-n_sample_retrain, ind_end=true_index)
                self.model = LGBMClassifier(
                    random_state=self.D_G.model_random_seed, verbose=-1)
                self.model.fit(X_train, y_train)

        self.errors = errors
        self.detection_indices = detection_indices
        self.retrain_name = retrain_name
        return (errors, detection_indices)

# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_adwin_loss(self, delta_w=0.01, delta_d=0.002, win_size=200, stop_first_detect=False):

        return (self.retrain_detector(detector_func=ADWIN,
                                      signal="log_loss",
                                      detector_params={
                                          "delta": delta_d, "clock": 1},
                                      warning_params={
                                          "delta": delta_w, "clock": 1},
                                      win_size=win_size,
                                      retrain_name="adwin_logloss",
                                      stop_first_detect=stop_first_detect))

    def retrain_adwin_point(self, delta_w=0.1, delta_d=0.01, win_size=200, clock=1, stop_first_detect=False):
        # TODO_ FIX THIS AS IT DOESNT WORK - SAME as other _point() functions
        return (self.retrain_detector(detector_func=ADWIN,
                                      signal="point",
                                      detector_params={
                                          "delta": delta_d, "clock": clock},
                                      warning_params={
                                          "delta": delta_w, "clock": clock},
                                      win_size=win_size,
                                      retrain_name="adwin_point",
                                      stop_first_detect=stop_first_detect))

    def retrain_adwin_shap(self, delta_w=0.01, delta_d=0.002, win_size=200, clock=1, bgd_type="sample", n_samp_bgd=25, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=ADWIN,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": delta_d, "clock": clock},
                                           warning_params={
                                               "delta": delta_w, "clock": clock},
                                           win_size=win_size,
                                           retrain_name="adwin_logloss_shap",
                                           bgd_type="sample",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))

    def retrain_PH_loss(self, delta_w=0.01, delta_d=0.005, win_size=200, min_instances=30, threshold=50.0, alpha=0.9999, stop_first_detect=False):
        return (self.retrain_detector(detector_func=PageHinkley,
                                      signal="log_loss",
                                      detector_params={
                                          "delta": delta_d,
                                          "threshold": threshold,
                                          "alpha": alpha,
                                          "min_instances": min_instances},
                                      warning_params={
                                          "delta": delta_w, },
                                      win_size=win_size,
                                      retrain_name="PH",
                                      stop_first_detect=stop_first_detect))

    def retrain_PH_loss(self, delta_w=0.01, delta_d=0.005, win_size=200, min_instances=30, threshold=50.0, alpha=0.9999, stop_first_detect=False):
        return (self.retrain_detector(detector_func=PageHinkley,
                                      signal="point",
                                      detector_params={
                                          "delta": delta_d,
                                          "threshold": threshold,
                                          "alpha": alpha,
                                          "min_instances": min_instances},
                                      warning_params={
                                          "delta": delta_w, },
                                      win_size=win_size,
                                      retrain_name="PH",
                                      stop_first_detect=stop_first_detect))

    # TODO: def retrain_PH_shap():
    def retrain_PH_shap(self, delta_w=0.01, delta_d=0.005, win_size=200, min_instances=30, threshold=50.0, alpha=0.9999, clock=1, bgd_type="sample",
                        n_samp_bgd=25, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": delta_d,
                                               "threshold": threshold,
                                               "alpha": alpha,
                                               "min_instances": min_instances},
                                           warning_params={"delta": delta_w},
                                           win_size=win_size,
                                           retrain_name="PH_logloss_shap",
                                           bgd_type="sample",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))

    # TODO: HDDM
    # TODO: def retrain_KSWIN_loss():
    def retrain_KSWIN_loss(self, win_size=200, alpha=0.005, w_size=100, stat_size=30, stop_first_detect=False):
        """

        TODO: fix error that happens; ValueError: object too deep for desired array
        """
        return (self.retrain_detector(detector_func=KSWIN,
                                      signal="log_loss",
                                      detector_params={"alpha": alpha,
                                                       "window_size": w_size,
                                                       "stat_size": stat_size},
                                      warning_params={},
                                      win_size=win_size,
                                      retrain_name="KSWIN_loss",
                                      stop_first_detect=stop_first_detect))

    # TODO: def retrain_KSWIN_loss():
    def retrain_KSWIN_point(self, win_size=200, alpha=0.005, w_size=100, stat_size=30, stop_first_detect=False):
        return (self.retrain_detector(detector_func=KSWIN,
                                      signal="point",
                                      detector_params={"alpha": alpha,
                                                       "window_size": w_size,
                                                       "stat_size": stat_size},
                                      warning_params={},
                                      win_size=win_size,
                                      retrain_name="KSWIN_point",
                                      stop_first_detect=stop_first_detect))

    def retrain_KSWIN_shap(self, win_size=200, alpha=0.005, w_size=100, stat_size=30,
                           clock=1, bgd_type="sample", n_samp_bgd=25, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=KSWIN,
                                           signal="log_loss",
                                           detector_params={"alpha": alpha,
                                                            "window_size": w_size,
                                                            "stat_size": stat_size},
                                           warning_params={},
                                           win_size=win_size,
                                           retrain_name="KSWIN_logloss_shap",
                                           bgd_type="sample",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))
# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_arf(self):
        """Adaptive Random forest"""
        model = AdaptiveRandomForestClassifier()
        X_train, y_train = self.D_G.get_x_y(ind_end=self.n_train)
        # X_test, y_test = self.D_G.get_x_y(ind_start=self.n_train, ind_end=self.n_train+self.n_test)
        model.fit(X_train.values, y_train)

        # init variables and returns
        errors = self.errors[:self.n_train+self.n_test]
        for x, label in zip(self.X_unseen.values, self.y_unseen):
            errors.append(int(model.predict([x]) == label))
            model.partial_fit([x], [label])

        self.errors = errors
        self.retrain_name = "arf"

        return (errors, [])


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def plot_retrain(self, detection_indices=None, w_size=None, ax=None, with_no_retrain=False):
        if (ax == None):
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 8])
        if (w_size == None):
            w_size = self.n_test

        n_seen = self.n_train+self.n_test  # number of sample used initially

        if (not self.classif):
            err_mean = self.get_current_mean_error(1)
        else:
            # compute traveling mean
            err_mean = self.get_current_mean_error(w_size)
        # plot error mean
        ax.plot(np.arange(n_seen, n_seen+len(err_mean)),
                err_mean, label="Mean error")
        ymax, ymin = max(err_mean[int(len(err_mean)*0.1):]
                         ), min(err_mean[int(len(err_mean)*0.1):])
        # ax.set_ylim(ymax=ymax+0.1*(ymax-ymin), ymin=ymin-0.1*(ymax-ymin))

        # plot detection lines
        ymin, ymax = ax.get_ylim()

        if (detection_indices == None and self.detection_indices != None):
            detection_indices = self.detection_indices
        if (detection_indices != None):
            if (with_no_retrain):
                if (not self.classif):
                    err_mean = self.get_original_mean_error(1)
                else:
                    err_mean = self.get_original_mean_error(w_size)

                # plot error mean
                ax.plot(np.arange(n_seen, n_seen+len(err_mean)), err_mean,
                        label="No retrain", color='grey', alpha=0.5, zorder=-99)

            for detect in detection_indices:
                # , label = "detection")
                ax.vlines(ymin=ymin, ymax=ymax, x=detect, color='orange')
            ax.scatter([], [], marker="|", color='orange',
                       label="detection", s=150)

        if (len(self.drift_points) > 0):
            for d in self.drift_points:
                ax.vlines(ymin=ymin, ymax=ymax, x=d,
                          color='crimson')
            ax.scatter([], [], marker="|", color='crimson',
                       label='drift', s=150)
        # ymin, ymax = ax.get_ylim()
        # plot training-test zone
        ax.axvspan(xmin=0, xmax=n_seen, alpha=0.5,
                   color='g', label="train-test")
        ax.set_title(self.drift_name+"\n"+self.retrain_name)
        ax.set_ylabel(f"mean error over {w_size} points")
        ax.set_xlabel(f"index")

        # LEGEND
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(
            zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

    def get_original_mean_error(self, w_size=None):
        if (w_size is None):
            w_size = self.n_test
        errors = self.D_G.errors[self.n_train:]
        err_mean = np.array([np.mean(errors[max(0, i-w_size):i])
                             for i in range(1, len(errors))][self.n_test:])
        return (err_mean)

    def get_current_mean_error(self, w_size=None):
        if (w_size is None):
            w_size = self.n_test
        errors = self.errors[self.n_train:]
        err_mean = np.array([np.mean(errors[max(0, i-w_size):i])
                             for i in range(1, len(errors))][self.n_test:])
        return (err_mean)

    # def get_current_abs_error(self, w_size=None):
    #    if (w_size is None):
    #        w_size = self.n_test
    #    errors = self.errors[self.n_train:]
    #    err_mean = np.array([np.mean(errors[max(0, i-w_size):i])
    #                         for i in range(1, len(errors))][self.n_test:])

    def get_changepoints(self, n_bkps=2):

        algo = rpt.Dynp(model="l2").fit(self.get_current_mean_error())
        result = algo.predict(n_bkps=n_bkps)
        return (result)
