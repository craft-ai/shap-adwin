import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib


from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMRegressor, LGBMClassifier
import math
import shap
import os
import pickle

from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import KSWIN, PageHinkley
# , KSWIN, PageHinkley


class Drift_generators():

    my_path = '/home/xxxxxxxxxxx/Documents/labs/data/results/'
    
    def __init__(self, n_samples=10**4, n_features=2):
        self.n = n_samples
        # if(n_features == 2):

        #    self.df = pd.DataFrame(columns =['x','y','class'])
        #    self.df['x'] = np.random.rand(n_samples)
        #    self.df['y'] = np.random.rand(n_samples)
        # else:
        self.df = pd.DataFrame(columns=['class'])
        for feat in range(n_features):
            self.df[feat] = np.random.rand(n_samples)
        self.n_features = n_features
        # apply default decision
        # hypersphere centered on 0.7,0.7,...,0.7 with default radius of 0.25
        self.default_decision = [0.4 for i in range(n_features)]

        self.n_train = min(1000, int(n_samples/10))
        self.n_test = int(0.2*self.n_train)
        self.drift_points = []
        self.drift_name = "NONE"
        self.model = None
        self.trained = False

    # def print_df_names(self):

    def load_df(self, df_name, n_train=None, n_test=None, nrows=60000):
        df = pd.read_csv("../data/"+df_name+".csv", nrows=nrows)
        self.n = len(df)
        if('class' in df.columns):
            if(not df.columns[-1] == 'class'):
                df.columns = np.concatenate(
                    [df.drop(columns='class').columns, ['class']])
            df.columns = ['class' if x == 'class' else i for i,
                          x in enumerate(df.columns)]

        self.df = df
        self.n_features = len(df.columns)-1
        if(n_train != None):
            self.n_train = n_train
        else:
            self.n_train = min(1000, int(self.n/10))
        if(n_test != None):
            self.n_test = n_test
        else:
            self.n_test = int(0.2*self.n_train)

        n_seen = self.n_train + self.n_test
        # train loaded DF
        X, y = self.df.loc[:n_seen].drop(
            columns=['class']), self.df.loc[:n_seen, 'class']
        X_train, y_train = X.loc[:self.n_train], y.loc[:self.n_train]
        X_test, y_test = X.loc[self.n_train:], y.loc[self.n_test:]
        # train initial model
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        #self.df = self.df.loc[n_seen:]
        self.model = model
        self.trained = True
        # check if we set drift points
        if(df_name in ["sine1", "sine2", "sine1_short", "sine2_short", "stagger", "stagger_short"]):
            if("short" in df_name):
                self.drift_points = [int(1/3*self.n), int(2/3*self.n)]
            elif("sine" in df_name):
                self.drift_points = [20020, 40030]
            elif("stagger" in df_name):
                self.drift_points = [33400, 66700]

    #    def set_drift_centers

    def set_default_decision(self, beg=None, end=None, decision=None):
        if(beg == None):
            beg = 0
        if(end == None):
            end = self.n
        if(decision != None):
            self.default_decision = decision
        self.df.loc[beg:end, 'class'] =\
            self.df.loc[beg:end].drop(columns=['class']).apply(
                lambda data: self.is_in_hypersphere(data.values, self.default_decision), axis=1)

    def is_in_hypersphere(self, point, center, r=0.25):
        if (self.n_features > 5):
            r = 0.4
        return(sum([(p-c)**2 for p, c in zip(point, center)]) < r**2)

    def brutal_concept_drift(self, n_drift=1, circle_centers=None, drift_points=None):
        """Simulate concept drift by arbitrarily changing the decision boundary"""
        if(circle_centers != None):
            if(len(circle_centers) == n_drift):
                circle_centers = [self.default_decision] + circle_centers
            elif(len(circle_centers) != n_drift+1):  # ELSE default case
                circle_centers = [
                    [0.7 for i in range(self.n_features)] for d in range(n_drift)]
        else:
            circle_centers = [[0.7 for i in range(self.n_features)] for d in range(
                n_drift)]  # drift circles

        if(drift_points == None):  # if no drift_point is specified just regularly drift along the dataset
            drift_points = [int(d*(self.n-self.n_train)/(n_drift+1)+self.n_train)
                            for d in range(1, n_drift+2)]
        else:
            if(len(drift_points) == n_drift):
                drift_points.append(self.n)
            elif(len(drift_points) < n_drift):
                drift_points = drift_points+[int(d * (self.n - drift_points[-1])/(n_drift+1-len(
                    drift_points))+drift_points[-1]) for d in range(len(drift_points), n_drift+1)]
            else:
                drift_points = [int(
                    d*(self.n-self.n_train)/(n_drift+1)+self.n_train) for d in range(1, n_drift+2)]

        self.set_default_decision(
            0, drift_points[0], decision=circle_centers[0])
        for d in range(n_drift):
            self.df.loc[drift_points[d]: drift_points[d+1], 'class'] =\
                self.df[drift_points[d]: drift_points[d+1]].drop(columns=['class']).apply(
                    lambda data: self.is_in_hypersphere(data.values, circle_centers[d+1]), axis=1)

        self.drift_points = drift_points[:-1]
        self.drift_name = "brutal concept drift"
        self.circle_centers = circle_centers

    def smooth_concept_drift(self, n_drift=1,  drift_points=None, circle_centers=None):
        """Simulate smooth concept drift by arbitrarily changing the decision boundary"""

        if(drift_points == None):  # if no drift_point is specified just regularly drift along the dataset
            drift_points = [int(d*(self.n-self.n_train)/(n_drift+1)+self.n_train)
                            for d in range(1, n_drift+2)]
        else:
            if(len(drift_points) == n_drift):
                drift_points.append(self.n)
            elif(len(drift_points) < n_drift):
                drift_points = drift_points+[int(d * (self.n - drift_points[-1])/(n_drift+1-len(
                    drift_points))+drift_points[-1]) for d in range(len(drift_points), n_drift+1)]
            else:
                drift_points = [int(
                    d*(self.n-self.n_train)/(n_drift+1)+self.n_train) for d in range(1, n_drift+2)]
        if(circle_centers == None):
            # circle_centers = [[0.3/10 for i in range(self.n_features)] for d in range(n_drift)]#drift circles
            circle_centers = [[0.5+(np.random.randint(3)-1)/8 for i in range(
                self.n_features)] for d in range(n_drift)]  # drift circles
        circle_centers = [self.default_decision] + circle_centers
        #print(drift_points, circle_centers)
        self.set_default_decision(0, drift_points[0])

        # first drift transition
        n_transi = drift_points[1] - drift_points[0]
        s_x = [(circle_centers[0][i]-self.default_decision[i]) /
               n_transi for i in range(self.n_features)]
        #print(drift_points[0], drift_points[1],"from",self.default_decision,"to",circle_centers[0],s_x)
        for k in range(n_transi):
            point = self.df.loc[drift_points[0] +
                                k].drop(columns=['class']).values
            center = [self.default_decision[i]+s_x[i]
                      * k for i in range(self.n_features)]
            #print(drift_points[0] + k, center)
            self.df.loc[drift_points[0] + k,
                        "class"] = self.is_in_hypersphere(point, center)

        # for several drifts
        for d in range(1, n_drift):
            n_transi = drift_points[d+1]-drift_points[d]
            s_x = [(circle_centers[d][i]-circle_centers[d-1][i]) /
                   n_transi for i in range(self.n_features)]
            # print(s_x)
            #print(d,drift_points[d], drift_points[d+1],"from", circle_centers[d-1]  ,"to",circle_centers[d],s_x)
            if(not np.any(s_x)):  # if the decision boundary does not change
                #    drift_points[d+1] = drift_points[d]
                #    print("no change")
                #    print(d,drift_points[d], drift_points[d+1],"from", circle_centers[d-1]  ,"to",circle_centers[d],s_x)
                self.set_default_decision(drift_points[d], drift_points[d+1])
                # self.df.loc[drift_points[d] : drift_points[d+1], 'class'] =\
                #    self.df[drift_points[d] : drift_points[d+1]].drop(columns=['class']).apply(lambda data: self.is_in_hypersphere(data.values, circle_centers[-1]), axis=1)
            # else:
            for k in range(n_transi):
                point = self.df.loc[drift_points[d] + k]
                center = [circle_centers[d-1][i]+s_x[i]
                          * k for i in range(self.n_features)]
                self.df.loc[drift_points[d] + k,
                            "class"] = self.is_in_hypersphere(point, center)
                #self.df.loc[drift_points[d] + k, "class"] = self.is_in_hypersphere(point, center)

        self.drift_points = drift_points[:-1]
        self.drift_name = "smooth concept drift"
        self.circle_centers = circle_centers

    def new_smooth_concept_drift(self, n_drift=1,  drift_points=None, circle_centers=None):
        """
            Smoothly change the decision frontier in bettween the drift points from one circle center to another
        """

        if(drift_points == None):  # if no drift_point is specified just regularly drift along the whole dataset
            drift_points = [int(d*(self.n-self.n_train)/(n_drift+1)+self.n_train)
                            for d in range(1, n_drift+2)]
        else:
            if(len(drift_points) == n_drift):
                drift_points.append(self.n)
            elif(len(drift_points) < n_drift):
                drift_points = drift_points+[int(d * (self.n - drift_points[-1])/(n_drift+1-len(
                    drift_points))+drift_points[-1]) for d in range(len(drift_points), n_drift+1)]

        if(circle_centers == None):  # randomly fix some circle_centers
            # circle_centers = [[0.3/10 for i in range(self.n_features)] for d in range(n_drift)]#drift circles
            circle_centers = [[0.5+(np.random.randint(3)-1)/8 for i in range(
                self.n_features)] for d in range(n_drift)]  # drift circles
        elif(len(circle_centers) == n_drift):
            # print("A")
            circle_centers = [self.default_decision] + circle_centers
        elif(len(circle_centers) > n_drift+1):
            circle_centers = circle_centers[:n_drift+1]
        elif(len(circle_centers) < n_drift):
            print("problem")
            #circle_centers = circle_centers[:n_drift+1]

        # print(drift_points,circle_centers)

        # set decision before drift
        self.set_default_decision(0, drift_points[0])
        # set decision during drift
        for d in range(n_drift):
            if(circle_centers[d+1] != circle_centers[d]):
                s_x = [(circle_centers[d+1][i]-circle_centers[d][i]) /
                       (drift_points[d+1]-drift_points[d]) for i in range(self.n_features)]
                for i, k in enumerate(range(drift_points[d], drift_points[d+1])):
                    point = self.df.drop(columns=['class']).loc[k].values
                    center = [(circle_centers[d][f]+s_x[f]*i)
                              for f in range(self.n_features)]
                    self.df.loc[k, 'class'] = self.is_in_hypersphere(
                        point, center)
            else:
                self.df.loc[drift_points[d]: drift_points[d+1], 'class'] =\
                    self.df.loc[drift_points[d]: drift_points[d+1]].drop(columns=['class']).apply(
                        lambda data: self.is_in_hypersphere(data.values, circle_centers[d+1]), axis=1)
        # set decision after drift as ended
        if(drift_points[-1] != self.n):
            self.df.loc[drift_points[-1]:, 'class'] =\
                self.df.loc[drift_points[-1]:].drop(columns=['class']).apply(
                    lambda data: self.is_in_hypersphere(data.values, circle_centers[-1]), axis=1)

        self.drift_points = drift_points  # [:-1]
        self.drift_name = "smooth concept drift"
        self.circle_centers = circle_centers

    def brutal_covariate_drift(self, d_centers=[('x', 0, 0.2, 0.8, 0.9), ('y', 0.5, 0.7, 0, 0.5)]):
        """
            The drift characteristics are defined the following way
            feature_drifting, start_point(as a % of len(df)), end_point(as a % of len(df)),
            min_feature_range(as a % of feat_values), max_feature range(as a % of feat_values)

            first make the feature drift
            then specify the decision boundary
        """

        drift_points = [
            int(x*self.n) for x in np.unique(np.ravel([(x[1], x[2]) for x in d_centers]))]
        # self.set_default_decision()

        for feat, beg, end, min_f, max_f in d_centers:
            beg, end = int(self.n*beg), int(self.n*end)
            self.df.loc[beg: end-1,
                        feat] = np.random.rand(end - beg)*(max_f-min_f) + min_f

        self.set_default_decision()

        self.drift_points = drift_points
        self.drift_name = "abrupt covariate"
        self.circle_centers = None

    def cyclic_abrupt_concept_drift(self, n_drift):
        circle_centers = [(0.5*np.cos(x), 0.5*np.sin(x))
                          for x in np.linspace(0, 2*np.pi, n_drift+1)]
        self.brutal_concept_drift(
            n_drift=n_drift, circle_centers=circle_centers)
        self.drift_name = "cyclic concept"

    def back_and_forth_abrupt_drift(self, circle_centers=None, drift_points=None):
        if(circle_centers == None):
            circle_centers = [
                [0.7 for i in range(self.n_features)], self.default_decision]
        self.brutal_concept_drift(
            n_drift=2, circle_centers=circle_centers, drift_points=drift_points)
        self.drift_name = "back and forth abrupt"

    def back_and_forth_smooth_drift(self):
        circle_centers = [[0.7 for i in range(
            self.n_features)], self.default_decision, self.default_decision]
        self.new_smooth_concept_drift(n_drift=3, circle_centers=circle_centers)
        self.drift_name = "back and forth smooth"

    def add_noise(self, noise_rate=0.05):
        # select and corrupt some rows labels
        noisy_rows = self.df.sample(int(self.n*noise_rate))
        self.df.loc[noisy_rows.index, 'class'] = noisy_rows.loc[:,
                                                                'class'].apply(lambda x: False if x else True)

    # def prior_proba_drift():
    #    return

    def get_first_detecs(self, detections, drift_points):
        """ 
        only look at the first detections
        We do not take into account redudant detections
        """
        if(len(drift_points) > 3):
            drift_points = [drift_points[0], drift_points[1], drift_points[-1]]

        detecs = np.array(detections, dtype=object)
        all_first_detecs = []
        # number of iterations
        for n in range(np.shape(detecs)[0]):
            # number of funcs
            detec_funcs = []
            for i in range(np.shape(detections)[1]):  # funcs
                first_detect = [[] for i in range(len(drift_points)-1)]
                for x in detecs[n, i]:
                    temp = list(drift_points)
                    temp.append(x)
                    temp = sorted(temp)
                    d_zone = temp.index(x)-1
                    if(len(first_detect[d_zone]) < 1):
                        first_detect[d_zone].append(x)
                detec_funcs.append(first_detect)
            all_first_detecs.append(detec_funcs)

        return(np.array(all_first_detecs, dtype=object))

    def get_detecs_and_drift_points(self, detections):

        detecs = np.array(detections, dtype=object)
        drift_points = np.unique(
            [0]+[x - self.n_seen() for x in self.drift_points]+[self.n-self.n_train-self.n_test])
        drift_points = [x for x in drift_points if 0 <=
                        x <= self.n-self.n_seen()]

        detecs = [[[x for x in detecs if drift_points[d_p] <= x < drift_points[d_p+1]]
                   for d_p in range(len(drift_points)-1)] for detecs in
                  [np.concatenate(np.array(detections, dtype=object)[:, i]).ravel()
                   for i in range(np.shape(detections)[1])]]

        return(detecs, drift_points)

    def get_stats_funcs(self, funcs, first_detecs):
        """
        This is designed for single_drift datasets
        """
        df_results = pd.DataFrame(columns=["freq_false", "freq_true",
                                           "mini", "mean", "std", "maxi", "median"])

        for i, func in enumerate(funcs):
            # print(func)
            detec_func = first_detecs[:, i]
            true_detecs = [x for x in detec_func[:, 1] if len(x) > 0]

            freq_false = len([x for x in detec_func[:, 0]
                             if len(x) > 0])/len(detec_func)
            freq_true = len([x for x in detec_func[:, 1]
                            if len(x) > 0])/len(detec_func)

            if(len(true_detecs) > 0):
                mean = int(np.mean(true_detecs))
                maxi = int(np.max(true_detecs))
                mini = int(np.min(true_detecs))
                median = int(np.median(true_detecs))
                std = np.std(true_detecs)
            else:
                mean = 0
                maxi = 0
                mini = 0
                median = 0
                std = 0

            df_results.loc[func.split("retrain_")[1]] = [
                freq_false, freq_true, mini, mean, std, maxi, median]
        return(df_results)

    def get_circles(self):
        return self.df

    def train(self, n_train=None, n_test=None):
        if(n_train != None):
            self.n_train = n_train
        if(n_test != None):
            self.n_test = n_test

        X, y = self.df.loc[:self.n_train+self.n_test].drop(
            columns=['class']), self.df.loc[:self.n_train+self.n_test, 'class'].astype(bool)
        X_train, y_train = X.loc[:self.n_train], y.loc[:self.n_train]
        X_test, y_test = X.loc[self.n_train:], y.loc[self.n_train:]
        # train initial model
        model = LGBMClassifier()
        model.fit(X_train, y_train)

        self.model = model
        self.score = model.score(X_test, y_test)
        self.trained = True
#        print("model trained, score on test set: %.3f"%self.score)

    def get_error(self):
        if(not self.trained):  # check whether a model was trained
            self.train()

        X_monitor = self.df.loc[self.n_train +
                                self.n_test:].drop(columns=['class'])
        y_monitor = self.df.loc[self.n_train +
                                self.n_test:, 'class'].astype(bool)
        retrain_indices = []

        # initialize error
        preds = self.model.predict(X_monitor.values)
        errors = [int(p == y) for p, y in zip(preds, y_monitor)]

        self.errors = errors
        return(errors, retrain_indices)

    def n_seen(self):
        return(self.n_train+self.n_test)

# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_DDM(self, w_threshold=2, d_threshold=3, n_cold_start=500):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_monitor = self.df.loc[self.n_train +
                                self.n_test:].drop(columns=['class'])
        y_monitor = self.df.loc[self.n_train +
                                self.n_test:, 'class'].astype(bool)
        # init variables and returns
        errors, error_mean, error_variance = [], [], []
        #index_first_detection = 0
        warning_indices = []
        detection_indices = []

        warning_threshold, detection_threshold = 1, 1
        e_count, count = 0, 0
        i, min_proba_err, min_var_err, first_warning = 0, 100, 100, np.inf

        while(i < len(X_monitor)):
            x, label = X_monitor.iloc[i], y_monitor.iloc[i]  # get next sample
            # update error and variance metrics ------------------
            error = int(model.predict([x]) == label)
            errors.append(error)
            if(error == 0):
                e_count += 1
            count += 1
            p_err = e_count/(count)
            error_mean.append(p_err)
            var_err = math.sqrt(p_err * (1 - p_err)/(count))
            error_variance.append(var_err)
            # -- detect drift-------------------------------------
            if(i > n_cold_start):  # number of cold start
                if((p_err + var_err) < (min_var_err + min_proba_err)):
                    min_proba_err, min_var_err = p_err, var_err
                warning_threshold = min_proba_err + w_threshold * min_var_err
                if((p_err + var_err) > warning_threshold):
                    if(count < first_warning):
                        first_warning = count
                    warning_indices.append(count)

                detection_threshold = min_proba_err + d_threshold * min_var_err
                if((p_err + var_err) > detection_threshold):
                    #print("DDEETTEECCTTIIOONN",count,"n_retrain : %d"%(count - first_warning))
                    detection_indices.append(count)
                    # train on the data since warning, if too fiew train on the 100 latest
                    n_samples_retrain = max(
                        n_cold_start, count - first_warning)
                    model = LGBMClassifier()
                    model.fit(
                        X_monitor.iloc[i-n_samples_retrain: i], y_monitor.iloc[i-n_samples_retrain: i])
                    # update the data we have not seen yet
                    X_monitor, y_monitor = X_monitor.iloc[i:], y_monitor.iloc[i:]
                    # reset DDM params
                    i, min_proba_err, min_var_err, first_warning = 0, 100, 100, np.inf

            i += 1
            # if(i%10000 == 0):
            #    print(i)
            #    print(i, "%.3f - %.3f - %.3f"%(warning_threshold, detection_threshold, p_err))

        self.errors = errors
        return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_periodic(self, period=1000, width=1000):
        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model
        if(width > period):  # maybe later add support for training set larger than re-training period
            width = period

        X_monitor = self.df.loc[self.n_train +
                                self.n_test:].drop(columns=['class'])
        y_monitor = self.df.loc[self.n_train +
                                self.n_test:, 'class'].astype(bool)

        # +D_G.n_train+D_G.n_test
        retrain_indices = [i for i in range(period, self.n, period)]

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_monitor.iloc[0:period]), y_monitor.iloc[0:period])]

        for i in range(period, len(X_monitor), period):
            # retrain
            model = LGBMClassifier()
            model.fit(X_monitor.iloc[max(0, i-width): i],
                      y_monitor.iloc[max(0, i-width): i])

            # compute errors
            preds = model.predict(X_monitor.iloc[max(0, i):i+period].values)
            errors += [int(p == y) for p, y in zip(preds,
                                                   y_monitor.iloc[max(0, i):min(len(y_monitor), i+period)])]

        self.errors = errors
        return(errors, retrain_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_periodic_increase(self, period=1000, width=500):
        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model
        if(width > period):  # maybe later add support for training set larger than re-training period
            width = period

        X_monitor = self.df.loc[self.n_train +
                                self.n_test:].drop(columns=['class'])
        y_monitor = self.df.loc[self.n_train +
                                self.n_test:, 'class'].astype(bool)
        # initialize train set
        X_train = self.df.loc[:self.n_train].drop(columns=['class'])
        y_train = self.df.loc[:self.n_train, 'class'].astype(bool)

        # +D_G.n_train+D_G.n_test
        retrain_indices = [i for i in range(period, self.n, period)]

        # initialize error
        preds = model.predict(X_monitor.iloc[0:period].values)
        errors = [int(p == y) for p, y in zip(preds, y_monitor.iloc[0:period])]

        for i in range(period, len(X_monitor), period):
            # retrain
            model = LGBMClassifier()

            X_train = pd.concat([X_train, X_monitor.iloc[max(0, i-width): i]])
            y_train = pd.concat([y_train, y_monitor.iloc[max(0, i-width): i]])
            model.fit(X_train, y_train)

            # compute errors
            preds = model.predict(X_monitor.iloc[max(0, i):i+period].values)
            errors += [int(p == y) for p, y in zip(preds,
                                                   y_monitor.iloc[max(0, i):min(len(y_monitor), i+period)])]

        self.errors = errors
        return(errors, retrain_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_feat_drift(self, win_size=100, test_prec=1e-10):
        '''
            retrain new model when a drift is detected in the one of the feature distributions.

            win_size: size of the windows to compute student test on.

        '''

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)

        errors = []
        detection_indices = []
        # drift detection
        # number of consecutive detection
        consec = [0 for x in range(self.n_features)]

        ref_win = [X_unseen.iloc[:win_size, feat]
                   for feat in range(self.n_features)]

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            errors.append(int(model.predict([x]) == label))

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            errors.append(int(model.predict([x]) == label))
            win = X_unseen.iloc[index-win_size:index]
            for feat in range(self.n_features):
                p_val = ttest_ind(win[feat], ref_win[feat])[1]
                if(p_val < test_prec):
                    consec[feat] += 1
                else:
                    consec[feat] = 0

                if(consec[feat] > win_size):
                    n_samples_retrain = min(index, win_size)

                    X_train = X_unseen.iloc[index - n_samples_retrain:index]
                    y_train = y_unseen.iloc[index - n_samples_retrain:index]

                    model = LGBMClassifier()
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    consec = [0 for x in range(self.n_features)]
                    ref_win = [X_train.iloc[:win_size, feat]
                               for feat in range(self.n_features)]

        self.errors = errors
        return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_feat_model_increase(self, win_size=100, test_prec=1e-10, n_estimators=50, n_esti_inc=5):
        # train initial model
        if(not self.trained):  # check whether a model was trained
            self.train()

        model = RandomForestClassifier(
            n_estimators=n_estimators)  # , warm_start=True)
        X_train = self.df.drop(columns='class').loc[:self.n_train]
        y_train = self.df.loc[:self.n_train, 'class']
        X_test = self.df.drop(
            columns='class').loc[self.n_train:self.n_train+self.n_test]
        y_test = self.df.loc[self.n_train:self.n_train+self.n_test, 'class']
        model.fit(X_train, y_train)

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)

        # prepare returns
        errors = []
        detection_indices = []
        # drift detection
        # number of consecutive detection
        consec = [0 for x in range(self.n_features)]
        # ind_detect = [0 for feat in range(self.n_features)]#can be used as warning index
        ref_win = [X_unseen.iloc[:win_size, feat]
                   for feat in range(self.n_features)]

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)

            for feat in range(self.n_features):
                win = X_unseen.iloc[index-win_size:index, feat]
                p_val = ttest_ind(win, ref_win[feat])[1]
                if(p_val < test_prec):
                    consec[feat] += 1
                    # if(ind_detect[feat] == 0):
                    #    ind_detect[feat] = index
                else:
                    consec[feat] = 0
                    #ind_detect[feat] = 0

                if(consec[feat] > win_size):  # train a single tree and add it to the ensemble
                    n_samples_retrain = min(index, win_size)

                    X_train = pd.concat(
                        [X_train, X_unseen.iloc[index - n_samples_retrain:index]])
                    y_train = pd.concat(
                        [y_train, y_unseen.iloc[index - n_samples_retrain:index]])

                    model = RandomForestClassifier(n_estimators=n_estimators)
                    #growing_rf.n_estimators += n_esti_inc
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    consec = [0 for x in range(self.n_features)]
                    ref_win = [X_train.iloc[:win_size, feat]
                               for feat in range(self.n_features)]

        self.errors = errors
        return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_basic_shap(self, win_size=500):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)
        # init shap explainer and reference window
        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(columns=['class'])
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            X_unseen[:win_size], y_unseen[:win_size])

        errors = []
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
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                X_unseen.iloc[[index]], y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                win = shap_values[index - win_size:index, feat]
                thresh = np.quantile(win, 0.75)
                win = list(filter(lambda x: x > thresh, win))
                p_val = ttest_ind(win, ref_win[feat])[1]
                if(p_val < 1e-5):
                    consec[feat] += 1
                    if(ind_detect[feat] == 0):
                        ind_detect[feat] = index
                else:
                    consec[feat] = 0
                    ind_detect[feat] = 0
                if(consec[feat] > 10):  # retrain
                    n_samples_retrain = min(index, win_size)
                    model = LGBMClassifier()
                    X_train = X_unseen.iloc[index - n_samples_retrain:index]
                    y_train = y_unseen.iloc[index - n_samples_retrain:index]
                    model.fit(X_train, y_train)

                    # reset detection
                    consec = [0 for x in range(self.n_features)]
                    detection_indices.append(index)
                    # recompute Shapley values
                    explainerError = shap.TreeExplainer(model, data=X_unseen.iloc[index - n_samples_retrain:index],
                                                        feature_perturbation="interventional", model_output='log_loss')
                    shap_values[index:index+win_size, :] = explainerError.shap_values(
                        X_unseen[index:index+win_size], y_unseen[index:index+win_size])
                    ref_win = [shap_values[index: index+win_size,  feat]
                               for feat in range(self.n_features)]
                    thresh = [np.quantile(ref, 0.75) for ref in ref_win]
                    ref_win = [list(filter(lambda x: x > thresh[i], ref))
                               for i, ref in enumerate(ref_win)]

        self.errors = errors
        return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_adwin(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)
        # init shap explainer and reference window
        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(
            columns=['class']).sample(min(50, self.n_train+self.n_test-1))
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            X_unseen[:win_size], y_unseen[:win_size])

        errors = []
        detection_indices = []
        # detection init
        warnings_feat = [ADWIN(beta_w) for feat in range(self.n_features)]
        detect_feat = [ADWIN(beta_d) for feat in range(self.n_features)]
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                X_unseen.iloc[[index]], y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])
                # print(shap_values[index,feat],feat)
                if warnings_feat[feat].detected_change():  # warning detection
                    if(first_warning > index):
                        first_warning = index
                if detect_feat[feat].detected_change():  # warning detection
                    n_samples_retrain = min(index, max(
                        index-first_warning, win_size))
                    # here we must fork wether we increase or retrain.
                    X_train = X_unseen.iloc[index - n_samples_retrain:index]
                    y_train = y_unseen.iloc[index - n_samples_retrain:index]

                    model = LGBMClassifier()
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    for feat in range(self.n_features):
                        warnings_feat[feat].reset()
                        detect_feat[feat].reset()
                    first_warning = np.inf
                    break

        self.errors = errors
        if(return_shap):
            return(errors, detection_indices, shap_values)
        else:
            return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_adwin_back_worse(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False):
        """
        shap explainer background data is composed of 50 points with the highest loss

        """

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)
        # init shap explainer and reference window
        X_test = self.df.loc[self.n_train:self.n_train +
                             self.n_test].drop(columns=['class'])
        y_test = self.df.loc[self.n_train:self.n_train +
                             self.n_test, 'class'].astype(bool)

        probas = [model.predict_proba([x])[0][1 if y else 0]
                  for x, y in zip(X_test.values, y_test)]
        loss = [-(label*np.log(proba))+(1-label)*np.log(1 - proba)
                for proba, label in zip(probas, y_test)]
       # X_test.iloc[[x[0] for x in sorted([(i,x) for i,x in enumerate(loss)], key=lambda x: x[1], reverse=True)][:50]]

        X_bgrd = X_test.iloc[[x[0] for x in sorted(
            [(i, x) for i, x in enumerate(loss)], key=lambda x: x[1], reverse=True)][:50]]
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            X_unseen[:win_size], y_unseen[:win_size])

        errors = []
        detection_indices = []
        # detection init
        warnings_feat = [ADWIN(beta_w) for feat in range(self.n_features)]
        detect_feat = [ADWIN(beta_d) for feat in range(self.n_features)]
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                X_unseen.iloc[[index]], y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])
                # print(shap_values[index,feat],feat)
                if warnings_feat[feat].detected_change():  # warning detection
                    if(first_warning > index):
                        first_warning = index
                if detect_feat[feat].detected_change():  # warning detection
                    n_samples_retrain = min(index, max(
                        index-first_warning, win_size))
                    # here we must fork wether we increase or retrain.
                    X_train = X_unseen.iloc[index - n_samples_retrain:index]
                    y_train = y_unseen.iloc[index - n_samples_retrain:index]

                    model = LGBMClassifier()
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    for feat in range(self.n_features):
                        warnings_feat[feat].reset()
                        detect_feat[feat].reset()
                    first_warning = np.inf
                    break

        self.errors = errors
        if(return_shap):
            return(errors, detection_indices, shap_values)
        else:
            return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_adwin_back_best(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False):
        """
            background is filled with with the 50 lowest loss points
        """
        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)
        # init shap explainer and reference window
        X_test = self.df.loc[self.n_train:self.n_train +
                             self.n_test].drop(columns=['class'])
        y_test = self.df.loc[self.n_train:self.n_train +
                             self.n_test, 'class'].astype(bool)

        probas = [model.predict_proba([x])[0][1 if y else 0]
                  for x, y in zip(X_test.values, y_test)]
        loss = [-(label*np.log(proba))+(1-label)*np.log(1 - proba)
                for proba, label in zip(probas, y_test)]
#        X_test.iloc[[x[0] for x in sorted([(i,x) for i,x in enumerate(loss)], key=lambda x: x[1], reverse=True)][50:]]

        X_bgrd = X_test.iloc[[x[0] for x in sorted(
            [(i, x) for i, x in enumerate(loss)], key=lambda x: x[1], reverse=True)][-50:]]
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            X_unseen[:win_size], y_unseen[:win_size])

        errors = []
        detection_indices = []
        # detection init
        warnings_feat = [ADWIN(beta_w) for feat in range(self.n_features)]
        detect_feat = [ADWIN(beta_d) for feat in range(self.n_features)]
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                X_unseen.iloc[[index]], y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])
                # print(shap_values[index,feat],feat)
                if warnings_feat[feat].detected_change():  # warning detection
                    if(first_warning > index):
                        first_warning = index
                if detect_feat[feat].detected_change():  # detection
                    n_samples_retrain = min(index, max(
                        index-first_warning, win_size))
                    # here we must fork wether we increase or retrain.
                    X_train = X_unseen.iloc[index - n_samples_retrain:index]
                    y_train = y_unseen.iloc[index - n_samples_retrain:index]

                    model = LGBMClassifier()
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    for feat in range(self.n_features):
                        warnings_feat[feat].reset()
                        detect_feat[feat].reset()
                    first_warning = np.inf
                    break

        self.errors = errors
        if(return_shap):
            return(errors, detection_indices, shap_values)
        else:
            return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_adwin_experimental(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)
        # init shap explainer and reference window
        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(
            columns=['class']).sample(min(50, self.n_train+self.n_test))
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            X_unseen[:win_size], y_unseen[:win_size])
        shap_init = shap_values[:win_size]

        errors = []
        detection_indices = []
        # detection init
        warnings_feat = [ADWIN(beta_w) for feat in range(self.n_features)]
        detect_feat = [ADWIN(beta_d) for feat in range(self.n_features)]
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                X_unseen.iloc[[index]], y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])
                # print(shap_values[index,feat],feat)
                if warnings_feat[feat].detected_change():  # warning detection
                    if(first_warning > index):
                        first_warning = index
                if detect_feat[feat].detected_change():  # drift detection
                    n_samples_retrain = min(index, max(
                        index-first_warning, win_size))
                    # here we must fork wether we increase or retrain.

                    shap_before_warn = shap_values[first_warning -
                                                   min(win_size, index):first_warning]
                    shap_problem = shap_values[index - n_samples_retrain:index]
                    #print("feature %d"%feat)

                    shaps = [x[:, feat]
                             for x in [shap_init, shap_before_warn, shap_problem]]
                    #print([np.shape(x) for x in shaps])
                    means = [(i, np.mean(x)) for i, x in enumerate(shaps)]
                    quantiles = [(i, np.quantile(x, 0.9))
                                 for i, x in enumerate(shaps)]
                    means = sorted(means, key=lambda x: x[1])
                    quantiles = sorted(quantiles, key=lambda x: x[1])
                    # print(means,quantiles)
                    print([x[0] for x in means].index(2), [x[0]
                          for x in quantiles].index(2))
                    if(([x[0] for x in means].index(2)+[x[0] for x in quantiles].index(2)) < 1):
                        print("THIS IS GETTING BETTER DO NOT INTERVENE !!!!",
                              index, n_samples_retrain)
                        for feat in range(self.n_features):
                            warnings_feat[feat].reset()
                            detect_feat[feat].reset()
                        first_warning = np.inf
                    else:
                        X_train = X_unseen.iloc[index -
                                                n_samples_retrain:index]
                        y_train = y_unseen.iloc[index -
                                                n_samples_retrain:index]

                        model = LGBMClassifier()
                        model.fit(X_train, y_train)

                        detection_indices.append(index)
                        for feat in range(self.n_features):
                            warnings_feat[feat].reset()
                            detect_feat[feat].reset()
                        first_warning = np.inf
                    break  # skip other features as we re-trained our model.

        self.errors = errors
        if(return_shap):
            return(errors, detection_indices, shap_values)
        else:
            return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_adwin_back_train(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)
        # init shap explainer and reference window
        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(columns=['class'])
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            X_unseen[:win_size], y_unseen[:win_size])

        errors = []
        detection_indices = []
        # detection init
        warnings_feat = [ADWIN(beta_w) for feat in range(self.n_features)]
        detect_feat = [ADWIN(beta_d) for feat in range(self.n_features)]
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                X_unseen.iloc[[index]], y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])
                # print(shap_values[index,feat],feat)
                if warnings_feat[feat].detected_change():  # warning detection
                    if(first_warning > index):
                        first_warning = index
                if detect_feat[feat].detected_change():  # warning detection
                    n_samples_retrain = min(index, max(
                        index-first_warning, win_size))

                    X_train = X_unseen.iloc[index - n_samples_retrain:index]
                    y_train = y_unseen.iloc[index - n_samples_retrain:index]

                    model = LGBMClassifier()
                    model.fit(X_train, y_train)

                    explainerError = shap.TreeExplainer(self.model, data=X_train,
                                                        feature_perturbation="interventional", model_output='log_loss')

                    detection_indices.append(index)
                    for feat in range(self.n_features):
                        warnings_feat[feat].reset()
                        detect_feat[feat].reset()
                    first_warning = np.inf
                    break

        self.errors = errors
        if(return_shap):
            return(errors, detection_indices, shap_values)
        else:
            return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_adwin_smallback(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)
        # init shap explainer and reference window
        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(columns=['class'])
        X_bgrd = X_bgrd.sample(2)
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            X_unseen[:win_size], y_unseen[:win_size])

        errors = []
        detection_indices = []
        # detection init
        warnings_feat = [ADWIN(beta_w) for feat in range(self.n_features)]
        detect_feat = [ADWIN(beta_d) for feat in range(self.n_features)]
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                X_unseen.iloc[[index]], y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])
                # print(shap_values[index,feat],feat)
                if warnings_feat[feat].detected_change():  # warning detection
                    if(first_warning > index):
                        first_warning = index
                if detect_feat[feat].detected_change():  # warning detection
                    n_samples_retrain = min(index, max(
                        index-first_warning, win_size))

                    X_train = X_unseen.iloc[index - n_samples_retrain:index]
                    y_train = y_unseen.iloc[index - n_samples_retrain:index]

                    model = LGBMClassifier()
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    for feat in range(self.n_features):
                        warnings_feat[feat].reset()
                        detect_feat[feat].reset()
                    first_warning = np.inf
                    break

        self.errors = errors
        if(return_shap):
            return(errors, detection_indices, shap_values)
        else:
            return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_adwin(self, beta_w=0.01, beta_d=0.002, win_size=200):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)

        errors = []
        detection_indices = []
        # detection init
        warning = ADWIN(beta_w)
        detect = ADWIN(beta_d)
        first_warning = np.inf

        for index in range(0, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]

            error = int(model.predict([x]) == label)
            errors.append(error)

            warning.add_element(error)
            detect.add_element(error)
            # print(shap_values[index,feat],feat)
            if warning.detected_change():  # warning detection
                if(first_warning > index):
                    first_warning = index
            if detect.detected_change():  # warning detection
                n_samples_retrain = min(index, max(
                    index-first_warning, win_size))

                X_train = X_unseen.iloc[index - n_samples_retrain:index]
                y_train = y_unseen.iloc[index - n_samples_retrain:index]

                model = LGBMClassifier()
                model.fit(X_train, y_train)

                detection_indices.append(index)
                for feat in range(self.n_features):
                    warning.reset()
                    detect.reset()
                first_warning = np.inf
                break

        self.errors = errors
        return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_adwin_stack_train(self, beta_w=0.01, beta_d=0.002, win_size=200):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)

        X_train = self.df.drop(columns='class').loc[:self.n_train]
        y_train = self.df.loc[:self.n_train, 'class']

        errors = []
        detection_indices = []
        # detection init
        warning = ADWIN(beta_w)
        detect = ADWIN(beta_d)
        first_warning = np.inf

        for index in range(0, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            proba = model.predict_proba([x])[0][1 if label else 0]
            loss = -(label*np.log(proba))+(1 - label)*np.log(1 - proba)
            error = int(model.predict([x]) == label)
            errors.append(error)

            warning.add_element(loss)
            detect.add_element(loss)
            # print(shap_values[index,feat],feat)
            if warning.detected_change():  # warning detection
                if(first_warning > index):
                    first_warning = index
            if detect.detected_change():  # warning detection
                n_samples_retrain = min(index, max(
                    index-first_warning, win_size))

                X_train = pd.concat(
                    [X_train, X_unseen.iloc[index - n_samples_retrain:index]])
                y_train = pd.concat(
                    [y_train, y_unseen.iloc[index - n_samples_retrain:index]])

                model = LGBMClassifier()
                model.fit(X_train, y_train)

                detection_indices.append(index)
                for feat in range(self.n_features):
                    warning.reset()
                    detect.reset()
                first_warning = np.inf
                break

        self.errors = errors
        # ------------------------------------------------------------------------------------------------------------------------------------------------
        return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_adwin_stack(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)
        X_train = self.df.drop(columns='class').loc[:self.n_train]
        y_train = self.df.loc[:self.n_train, 'class']
        # init shap explainer and reference window
        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(
            columns=['class']).sample(min(50, self.n_train+self.n_test))
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            X_unseen[:win_size], y_unseen[:win_size])

        errors = []
        detection_indices = []
        # detection init
        warnings_feat = [ADWIN(beta_w) for feat in range(self.n_features)]
        detect_feat = [ADWIN(beta_d) for feat in range(self.n_features)]
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                X_unseen.iloc[[index]], y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])
                # print(shap_values[index,feat],feat)
                if warnings_feat[feat].detected_change():  # warning detection
                    if(first_warning > index):
                        first_warning = index
                if detect_feat[feat].detected_change():  # warning detection
                    n_samples_retrain = min(index, max(
                        index-first_warning, win_size))

                    X_train = pd.concat(
                        [X_train, X_unseen.iloc[index - n_samples_retrain:index]])
                    y_train = pd.concat(
                        [y_train, y_unseen.iloc[index - n_samples_retrain:index]])

                    model = LGBMClassifier()
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    for feat in range(self.n_features):
                        warnings_feat[feat].reset()
                        detect_feat[feat].reset()
                    first_warning = np.inf
                    break

        self.errors = errors
        if(return_shap):
            return(errors, detection_indices, shap_values)
        else:
            return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_KSWIN(self, alpha_w=0.01, alpha_d=0.002, win_size=200, return_shap=False):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)
        # init shap explainer and reference window
        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(columns=['class'])
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            X_unseen[:win_size], y_unseen[:win_size])

        errors = []
        detection_indices = []
        # detection init
        warnings_feat = [KSWIN(alpha_w) for feat in range(self.n_features)]
        detect_feat = [KSWIN(alpha_d) for feat in range(self.n_features)]
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                X_unseen.iloc[[index]], y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])
                # print(shap_values[index,feat],feat)
                if warnings_feat[feat].detected_change():  # warning detection
                    if(first_warning > index):
                        first_warning = index
                if detect_feat[feat].detected_change():  # warning detection
                    n_samples_retrain = min(index, max(
                        index-first_warning, win_size))

                    X_train = X_unseen.iloc[index - n_samples_retrain:index]
                    y_train = y_unseen.iloc[index - n_samples_retrain:index]

                    model = LGBMClassifier()
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    for feat in range(self.n_features):
                        warnings_feat[feat].reset()
                        detect_feat[feat].reset()
                    first_warning = np.inf
                    break

        self.errors = errors
        if(return_shap):
            return(errors, detection_indices, shap_values)
        else:
            return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_KSWIN(self, alpha_w=0.01, alpha_d=0.002, win_size=200):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)

        errors = []
        detection_indices = []
        # detection init
        warning = KSWIN(alpha_w)
        detect = KSWIN(alpha_d)
        first_warning = np.inf

        for index in range(0, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            proba = model.predict_proba([x])[0][1 if label else 0]
            loss = -(label*np.log(proba))+(1 - label)*np.log(1 - proba)
            error = int(model.predict([x]) == label)
            errors.append(error)

            warning.add_element(loss)
            detect.add_element(loss)
            # print(shap_values[index,feat],feat)
            if warning.detected_change():  # warning detection
                if(first_warning > index):
                    first_warning = index
            if detect.detected_change():  # warning detection
                n_samples_retrain = min(index, max(
                    index-first_warning, win_size))

                X_train = X_unseen.iloc[index - n_samples_retrain:index]
                y_train = y_unseen.iloc[index - n_samples_retrain:index]

                model = LGBMClassifier()
                model.fit(X_train, y_train)

                detection_indices.append(index)
                for feat in range(self.n_features):
                    warning.reset()
                    detect.reset()
                first_warning = np.inf

        self.errors = errors
        return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_PH(self, delta_w=0.01, delta_d=0.005, win_size=200, return_shap=False):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)
        # init shap explainer and reference window
        X_bgrd = self.df.loc[:self.n_train+self.n_test].drop(columns=['class'])
        explainerError = shap.TreeExplainer(self.model, data=X_bgrd,
                                            feature_perturbation="interventional", model_output='log_loss')
        shap_values = np.zeros((len(X_unseen), self.n_features))
        shap_values[:win_size] = explainerError.shap_values(
            X_unseen[:win_size], y_unseen[:win_size])

        errors = []
        detection_indices = []
        # detection init
        warnings_feat = [PageHinkley(delta_w)
                         for feat in range(self.n_features)]
        detect_feat = [PageHinkley(delta_d) for feat in range(self.n_features)]
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            error = int(model.predict([x]) == label)
            errors.append(error)
            shap_values[index] = explainerError.shap_values(
                X_unseen.iloc[[index]], y_unseen.iloc[[index]])[0]

            for feat in range(self.n_features):
                warnings_feat[feat].add_element(shap_values[index, feat])
                detect_feat[feat].add_element(shap_values[index, feat])
                # print(shap_values[index,feat],feat)
                if warnings_feat[feat].detected_change():  # warning detection
                    if(first_warning > index):
                        first_warning = index
                if detect_feat[feat].detected_change():  # warning detection
                    n_samples_retrain = min(index, max(
                        index-first_warning, win_size))

                    X_train = X_unseen.iloc[index - n_samples_retrain:index]
                    y_train = y_unseen.iloc[index - n_samples_retrain:index]

                    model = LGBMClassifier()
                    model.fit(X_train, y_train)

                    detection_indices.append(index)
                    for feat in range(self.n_features):
                        warnings_feat[feat].reset()
                        detect_feat[feat].reset()
                    first_warning = np.inf
                    break

        self.errors = errors
        if(return_shap):
            return(errors, detection_indices, shap_values)
        else:
            return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_PH(self, delta_w=0.01, delta_d=0.005, win_size=200):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)

        errors = []
        detection_indices = []
        # detection init
        warning = PageHinkley(delta_w)
        detect = PageHinkley(delta_d)
        first_warning = np.inf

        for index in range(0, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            proba = model.predict_proba([x])[0][1 if label else 0]
            loss = -(label*np.log(proba))+(1 - label)*np.log(1 - proba)
            error = int(model.predict([x]) == label)
            errors.append(error)

            warning.add_element(loss)
            detect.add_element(loss)
            # print(shap_values[index,feat],feat)
            if warning.detected_change():  # warning detection
                if(first_warning > index):
                    first_warning = index
            if detect.detected_change():  # warning detection
                n_samples_retrain = min(index, max(
                    index-first_warning, win_size))

                X_train = X_unseen.iloc[index - n_samples_retrain:index]
                y_train = y_unseen.iloc[index - n_samples_retrain:index]

                model = LGBMClassifier()
                model.fit(X_train, y_train)

                detection_indices.append(index)
                for feat in range(self.n_features):
                    warning.reset()
                    detect.reset()
                first_warning = np.inf

        self.errors = errors
        return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_adwin_loss(self, beta_w=0.01, beta_d=0.002, win_size=200):

        if(not self.trained):  # check whether a model was trained
            self.train()
        model = self.model

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)

        errors = []
        detection_indices = []
        # detection init
        warning = ADWIN(beta_w)
        detect = ADWIN(beta_d)
        first_warning = np.inf

        # initialize error
        errors = [int(pred == label) for pred, label in zip(
            model.predict(X_unseen.iloc[0:win_size]), y_unseen.iloc[0:win_size])]

        # initialize detectors
        for index in range(win_size):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]
            proba = model.predict_proba([x])[0][1 if label else 0]
            loss = -(label*np.log(proba))+(1 - label)*np.log(1 - proba)
            warning.add_element(loss)
            detect.add_element(loss)

        for index in range(win_size, len(X_unseen)):
            # get next sample
            x, label = X_unseen.iloc[index], y_unseen.iloc[index]

            proba = model.predict_proba([x])[0][1 if label else 0]
            loss = -(label*np.log(proba))+(1 - label)*np.log(1 - proba)
            # print(model.predict_proba([x]))
            error = int(model.predict([x]) == label)
            errors.append(error)

            warning.add_element(loss)
            detect.add_element(loss)
            # print(shap_values[index,feat],feat)
            if warning.detected_change():  # warning detection
                if(first_warning > index):
                    first_warning = index
            if detect.detected_change():  # warning detection
                n_samples_retrain = min(index, max(
                    index-first_warning, win_size))

                X_train = X_unseen.iloc[index - n_samples_retrain:index]
                y_train = y_unseen.iloc[index - n_samples_retrain:index]

                model = LGBMClassifier()
                model.fit(X_train, y_train)

                detection_indices.append(index)
                for feat in range(self.n_features):
                    warning.reset()
                    detect.reset()
                first_warning = np.inf

        self.errors = errors
        return(errors, detection_indices)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_arf(self):
        """Adaptive Random forest"""
        if(not self.trained):  # check whether a model was trained
            self.train()
        model = AdaptiveRandomForestClassifier()
        X_train = self.df.drop(columns='class').loc[:self.n_train]
        y_train = self.df.loc[:self.n_train, 'class']
        X_test = self.df.drop(
            columns='class').loc[self.n_train:self.n_train+self.n_test]
        y_test = self.df.loc[self.n_train:self.n_train+self.n_test, 'class']
        model.fit(X_train.values, y_train)

        X_unseen = self.df.loc[self.n_train +
                               self.n_test:].drop(columns=['class'])
        y_unseen = self.df.loc[self.n_train+self.n_test:, 'class'].astype(bool)
        # init variables and returns
        errors = []
        for x, label in zip(X_unseen.values, y_unseen):
            errors.append(int(model.predict([x]) == label))
            model.partial_fit([x], [label])

        self.errors = errors
        self.drift_name = "arf"

        return(errors, [])


# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_nothing(self):
        self.drift_name = "nothing"
        return(self.get_error())
    # def retrain_DDM_increase(self, period = 1000, width = 500):
        # TBD
    #    return(errors, error_mean, retrain_indices)

# ----------------------PLOTS--------------------
    def pre_compute_plot(self, error_list, detections, funcs, w_size=500):
        # pre-compute error means
        error_means_list = [[] for i in range(len(funcs))]

        for k in range(np.shape(error_list)[0]):
            for i, (err, label) in enumerate(zip(error_list[k], funcs)):
                if(len(err) > 0):
                    error_mean = [np.mean(err[max(0, i-w_size):i])
                                  for i in range(1, len(err))]
                    error_means_list[i].append(error_mean)

        # pre-compute DETECTION METRICS
        detecs = np.array(detections, dtype=object)

        drift_points = np.unique(
            [0]+[x - self.n_seen() for x in self.drift_points]+[self.n-self.n_train-self.n_test])
        drift_points = [x for x in drift_points if 0 <=
                        x <= self.n-self.n_seen()]

        freq_detections = [np.mean([[1 if x >= 1 else 0 for x in
                                     [len([x for x in detecs[k, i] if drift_points[d_p] <= x < drift_points[d_p+1]])
                                      for d_p in range(len(drift_points)-1)]]
                                    for k in range(np.shape(error_list)[0])], axis=0)
                           for i in range(len(funcs))]

        stats_detections = [[] for i in range(len(funcs))]
        for i in range(len(funcs)):
            detection_func = np.concatenate(detecs[:, i]).ravel()

            stats_func = [[] for i in range(len(drift_points)-1)]
            for d_p in range(len(drift_points)-1):
                detections_interval = [
                    x for x in detection_func if drift_points[d_p] <= x < drift_points[d_p+1]]
                if(len(detections_interval) > 0):
                    stats_func[d_p] = [int(np.mean(detections_interval)),
                                       int(np.std(detections_interval)),
                                       len(detections_interval),
                                       min(detections_interval)]
                else:
                    stats_func[d_p] = [0, 0, 0, np.nan]

            stats_detections[i] = stats_func

        detecs = [[[x for x in detecs if drift_points[d_p] <= x < drift_points[d_p+1]]
                   for d_p in range(len(drift_points)-1)] for detecs in
                  [np.concatenate(np.array(detections, dtype=object)[:, i]).ravel()
                   for i in range(len(funcs))]]

        return(error_means_list, drift_points, detecs, stats_detections, freq_detections)

    def plot_retrain(self, detection_indices=[], w_size=500, ax=None):
        if(ax == None):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 8])

        n_seen = self.n_train+self.n_test  # number of sample used initially
        # compute traveling mean
        err_mean = [np.mean(self.errors[max(0, i-w_size):i])
                    for i in range(1, len(self.errors))]
        # plot error mean
        ax.plot(np.arange(n_seen, n_seen+len(err_mean)),
                err_mean, label="Mean error")
        ymax, ymin = max(err_mean[int(len(err_mean)*0.1):]
                         ), min(err_mean[int(len(err_mean)*0.1):])
        ax.set_ylim(ymax=ymax+0.1*(ymax-ymin), ymin=ymin-0.1*(ymax-ymin))
        # plot detection lines
        ymin, ymax = ax.get_ylim()
        for detect in detection_indices:
            ax.vlines(ymin=ymin, ymax=ymax, x=detect+n_seen,
                      color='orange', label="detection")
        for d in self.drift_points:
            ax.vlines(ymin=ymin, ymax=ymax, x=d,
                      color='crimson', label='drift')
        #ymin, ymax = ax.get_ylim()
        # plot training-test zone
        ax.axvspan(xmin=0, xmax=n_seen, alpha=0.5,
                   color='g', label="train-test")
        ax.set_title(self.drift_name+"\nmean error")
        # LEGEND
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(
            zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

    def plot_circles(self):
        if (self.n_features == 2):
            cmap = matplotlib.cm.get_cmap('tab10')

            fix, ax = plt.subplots(nrows=1, ncols=1, figsize=[20, 10])
            # drift_centers):
            for i, (c_x, c_y) in enumerate(self.circle_centers):
                circle = plt.Circle((c_x, c_y), 0.25, color=cmap(
                    i), fill=False, label="drift_"+str(i))
                ax.add_patch(circle)

            c_x, c_y = self.default_decision
            # for (c_x, c_y) in self.default_decision:# drift_centers):
            circle = plt.Circle((c_x, c_y), 0.25, color='green',
                                fill=False, label="default decision")
            ax.add_patch(circle)
        # can be replaced with a proper setting of plot frame
            ax.scatter([x[0] for x in self.circle_centers], [x[1]
                       for x in self.circle_centers], marker='x')
            ax.legend()
            ax.set_title(self.drift_name+"\nmean error")
            plt.show()
        else:
            print("too many features to plot maybe PCA to implement later")

    def plot_compare_detects_old(self, funcs, look_funcs, drift_points, stats_detections, freq_detections, ax=None):

        # filter elts we need and order results
        i_selected = [funcs.index(x) for x in look_funcs if x in funcs]
        fcts = np.array(funcs)[i_selected]
        stats = np.array(stats_detections)[i_selected]
        # first order funcs by mean detec - can represent the function rank
        o_funcs = [x[0] for x in sorted(
            [(func, stats[i][1][0]) for i, func in enumerate(fcts)], key=lambda x:x[1], reverse=True)]
        i_ordered = [list(fcts).index(x) for x in o_funcs if x in fcts]
        stats = np.array(stats)[i_ordered]
        freq_detec_ordered = np.array(freq_detections)[i_selected][i_ordered]

        if(ax == None):
            fig, ax = plt.subplots(
                nrows=len(drift_points)-1, ncols=1, figsize=[20, 2*len(fcts)])

        colors = ['b', 'g', 'purple', 'cyan',
                  'lime', 'palegoldenrod', 'navy', 'flame']
        if(len(funcs) > 8):
            colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
            by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                            for name, color in colors.items())
            colors = [name for hsv, name in by_hsv]
            np.random.shuffle(colors)

        ymin_max = [(i)/len(fcts) for i in range(len(fcts)+1)]

        for d, d_p in enumerate(drift_points[:-1]):
            ax[d].vlines(ymin=0, ymax=1, x=d_p + self.n_seen(),
                         color='red', label="drift point")

        for i, func in enumerate(o_funcs):
            for d, d_p in enumerate(drift_points[:-1]):
                ax[d].set_xlim(xmin=d_p + self.n_seen()-100,
                               xmax=drift_points[d+1] + self.n_seen()-100)
                mean, var, count, early = stats[i][d]
                rank = o_funcs.index(func)
                ymin, ymax = ymin_max[rank:rank+2]
                if(mean > 0):

                    ax[d].vlines(ymin=ymin, ymax=ymax, x=mean+self.n_seen(),
                                 color=colors[i], label="%.2f : %s | + %d" % (freq_detec_ordered[i][d], func[8:], int(mean-d_p)))
                    ax[d].fill_betweenx(y=[ymin, ymax], x1=max(early, mean - var)+self.n_seen(),
                                        x2=min(drift_points[d+1], mean + var)+self.n_seen(), alpha=0.35, color=colors[i])
                else:
                    ax[d].plot([], [], ' ', label="0 : "+func[8:])
                if(early > 0):
                    ax[d].vlines(ymin=ymin, ymax=ymax, x=early + self.n_seen(),
                                 color=colors[i])
        for d in range(len(drift_points[:-1])):
            ax[d].set_title("Drift n %d %s" % (d, " "*30))
            handles, labels = ax[d].get_legend_handles_labels()
            by_label = dict(zip(reversed(labels), reversed(handles)))
            ax[d].legend(by_label.values(), by_label.keys(), fontsize=12)

    def plot_compare_detects(self, funcs, look_funcs, drift_points, stats_detections, freq_detections, detecs, ax=None):

        # filter elts we need and order results
        i_selected = [funcs.index(x) for x in look_funcs if x in funcs]
        fcts = np.array(funcs)[i_selected]
        stats = np.array(stats_detections)[i_selected]
        # first order funcs by mean detec - can represent the function rank
        o_funcs = [x[0] for x in sorted(
            [(func, stats[i][1][0]) for i, func in enumerate(fcts)], key=lambda x:x[1], reverse=True)]
        i_ordered = [list(fcts).index(x) for x in o_funcs if x in fcts]
        stats = np.array(stats)[i_ordered]
        freq_detec_ordered = np.array(freq_detections)[i_selected][i_ordered]
        o_detecs = np.array(detecs, dtype='object')[i_selected][i_ordered]

        ax_sizes = [sum([len(x) > 0 for x in o_detecs[:, i]])
                    for i in range(np.shape(o_detecs)[1])]
        ax_sizes = [x+1 if x == 0 else x for x in ax_sizes]

        if(ax == None):
            fig, ax = plt.subplots(nrows=np.shape(o_detecs)[1], ncols=1,
                                   figsize=[20, sum(ax_sizes)*0.75], gridspec_kw={'height_ratios': ax_sizes})

        colors = ['orange', 'cyan', 'crimson', 'lime', 'violet', 'dodgerblue',
                  'lightslategrey', 'olivedrab']
        if(len(funcs) > 8):
            colors = [[x for x in colors][i %
                                          len(colors)] for i in range(len(funcs))]

        ymin_max = [(i)/len(fcts) for i in range(len(fcts)+1)]

        for d, d_p in enumerate(drift_points[:-1]):
            ax[d].vlines(ymin=0, ymax=1, x=d_p,
                         color='red', label="drift point")
            ax[d].vlines(ymin=0, ymax=1, x=drift_points[d+1],
                         color='red', label="drift point")

        for d, d_p in enumerate(drift_points[:-1]):

            index_detecs = [i for i, x in enumerate(
                freq_detec_ordered[:, d]) if x > 0]

            if(len(index_detecs) > 0):
                func_detecs = o_detecs[index_detecs, d]

                violins = ax[d].violinplot(func_detecs, vert=False, showmeans=True,
                                           widths=1.2/(ax_sizes[d]+1),
                                           positions=[(i)/(ax_sizes[d]+1) for i in range(1, ax_sizes[d]+1)])
                # for ((i,func), v) in zip(enumerate(o_funcs),violins['bodies']):
                for i, (ind, func, v) in enumerate(zip(index_detecs, np.array(o_funcs)[index_detecs], violins['bodies'])):
                    v.set_fc(c=colors[ind])
                    if(freq_detec_ordered[i][d] < 0.98):
                        v.set_alpha(0.33)
                    else:
                        v.set_alpha(1)

                    ax[d].plot([], [], color=colors[ind],
                               label="%.2f : %s | + %d" % (freq_detec_ordered[ind, d],
                                                           func[8:], violins['cmeans'].get_segments()[i][0][0] - drift_points[d]))

                violins['cmaxes'].set_edgecolor('red')
                violins['cmins'].set_edgecolor('red')

        for d in range(len(drift_points[:-1])):
            ax[d].set_title("Drift n %d %s" % (d, " "*30))
            handles, labels = ax[d].get_legend_handles_labels()
            by_label = dict(zip(reversed(labels), reversed(handles)))
            ax[d].legend(by_label.values(), by_label.keys(), fontsize=12)
            interval = drift_points[d+1]-drift_points[d]
            ax[d].set_xlim(xmin=drift_points[d]-interval/10,
                           xmax=drift_points[d+1]+interval/3)
            ax[d].set_ylim(ymin=0, ymax=1)

    def plot_mean_results(self, funcs, error_means_list, drift_points, stats_detections, freq_detections, ax=None, ymin=None, ymax=None):

        if(ax == None):
            fig, ax = plt.subplots(
                nrows=len(funcs), ncols=1, figsize=[20, 3*len(funcs)])

        for i, func in enumerate(funcs):
            mean = np.mean(error_means_list[i], axis=0)
            var = np.std(error_means_list[i], axis=0)
            ax[i].plot(mean, label="mean-error")
            ax[i].fill_between(np.arange(len(mean)), mean -
                               var, mean+var, alpha=0.30)
            ax[i].set_title("%.3f  " % (np.mean(mean))+func[8:], fontsize=20)

            if(ymin == None):
                ymin = 0.85
            if(ymax == None):
                ymax = 1

            for d in self.drift_points:
                ax[i].vlines(ymin=ymin, ymax=ymax, x=d -
                             self.n_seen(), color='red', label="drift point")

            for d_p in range(len(drift_points)-1):
                mean, var, count, early = stats_detections[i][d_p]
                if(mean != 0):
                    ax[i].vlines(ymin=ymin, ymax=ymax, x=mean,
                                 color='orange', label="mean detec")
                    ax[i].vlines(ymin=ymin, ymax=ymax, x=early,
                                 color='lime', label="earliest detec")
                    ax[i].fill_betweenx(y=[0, 1], x1=max(early, mean - var),
                                        x2=min(drift_points[d_p+1], mean + var), alpha=0.35, color='orange')

            # plot detections
            ax[i].plot([], [], ' ',
                       label="detec freqs: "+' | '.join('%.2f' % k for k in freq_detections[i]))
            ax[i].set_ylim(ymin, ymax)
            ax[i].set_xticks([])
            handles, labels = ax[i].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax[i].legend(by_label.values(), by_label.keys(),
                         fontsize=12, loc="lower left")

    def plot_every_iteration(self, funcs, error_means_list, detections, ymin=0.5, ymax=1, ax=None):

        if(ax == None):
            fig, ax = plt.subplots(
                nrows=len(funcs), ncols=1, figsize=[20, 4*len(funcs)])
        detections = np.array(detections, dtype=object)
        n_seen = self.n_train+self.n_test

        for k in range(np.shape(error_means_list)[1]):
            for i, label in enumerate(funcs):
                error_mean = error_means_list[i][k]
                ax[i].plot(np.arange(n_seen, n_seen +
                           len(error_mean)), error_mean)
                ax[i].set_ylim(ymin, ymax)

                for d in detections[k, i]:
                    ax[i].vlines(ymin=ymin, ymax=ymax, x=d +
                                 n_seen, color='orange', alpha=0.25)

        for i, func in enumerate(funcs):
            ax[i].set_title(func[8:]+"\nerror: %.2f - %d detections" % (np.mean(error_means_list[i]),
                            len(np.concatenate(detections[:, i]).ravel())/detections.shape[0]),
                            fontsize=15)
            ax[i].set_xticks([])

    def plot_every_iteration_new(self, funcs, look_funcs, error_means_list, detections, ymin=0.5, ymax=1, ax=None):

        i_selected = [funcs.index(x) for x in look_funcs if x in funcs]
        fcts = np.array(funcs)[i_selected]

        error_means_list = np.array(error_means_list)[i_selected]
        detections = np.transpose(np.transpose(
            np.array(detections, dtype=object))[i_selected])

        if(ax == None):
            fig, ax = plt.subplots(
                nrows=len(fcts), ncols=1, figsize=[20, 4*len(fcts)])
        detections = np.array(detections, dtype=object)
        n_seen = self.n_train+self.n_test

        for k in range(np.shape(error_means_list)[1]):
            for i, label in enumerate(fcts):
                error_mean = error_means_list[i][k]
                ax[i].plot(np.arange(n_seen, n_seen +
                           len(error_mean)), error_mean)
                ax[i].set_ylim(ymin, ymax)

                for d in detections[k, i]:
                    ax[i].vlines(ymin=ymin, ymax=ymax, x=d +
                                 n_seen, color='orange', alpha=0.25)

        for i, func in enumerate(fcts):
            ax[i].set_title(func[8:]+"\nerror: %.2f - %d detections" % (np.mean(error_means_list[i]),
                            len(np.concatenate(detections[:, i]).ravel())/detections.shape[0]),
                            fontsize=15)
            ax[i].set_xticks([])

    def plot_stack_mean(self, funcs, error_means_list, ymin=0.5, ymax=1, ax=None):

        if(ax == None):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[20, 12])

        for i, func in enumerate(funcs):
            if(func == "retrain_shap_adwin"):
                func = "retrain_shap_adwin_random"
            mean = np.mean(error_means_list[i], axis=0)
            var = np.std(error_means_list[i], axis=0)
            ax.plot(mean, label="%.3f  " % np.mean(mean)+func[8:])
            ax.fill_between(np.arange(len(mean)), mean -
                            var, mean+var, alpha=0.30)

            ax.set_ylim(ymin, ymax)

        # sort legend
        labels = ax.get_legend_handles_labels()
        labels = [(a, b, float(b[:5])) for a, b in zip(labels[0], labels[1])]
        labels = sorted(labels, key=lambda x: x[2], reverse=True)
        handles, labels = [x[0] for x in labels], [x[1] for x in labels]

        plt.legend(handles, labels, fontsize=20)

# _---------------------------------LOAD------------------

    def load_data(self, funcs, dataset="brutal_concept_backforth", n_load=10):

        print([x[8:] for x in funcs])

        n_iter = min(n_load, min([len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                                 for path in [Drift_generators.my_path+dataset+"/"+func[8:] for func in funcs]]))

        error_list = [[[] for i in range(len(funcs))] for k in range(n_iter)]
        detections = [[[] for i in range(len(funcs))] for k in range(n_iter)]

        for i, func in enumerate(funcs):
            path = Drift_generators.my_path + "/" + \
                dataset+"/"+func[8:]
            path = path.replace("//","/")
            path = path.replace("//","/")
            files = np.sort([f for f in os.listdir(
                path) if os.path.isfile(os.path.join(path, f))])[-n_iter:]
            for k, f in enumerate(files):
                err, detec = pickle.load(open(path+"/"+f, 'rb'))
                error_list[k][i] = err
                detections[k][i] = detec
        print("\n%d results loaded for each policy (%d funcs)" %
              (n_iter, len(funcs)))
        return(error_list, detections)


# _---------------------------------PERF------------------

    def compute_retrain_perf(self, errors):
        """
            Not checked yet
        """
    #    error_measure_corrected_by_variance =  np.asarray([math.sqrt(val * (1 - val) / (ix +1)) for
    #                                                         ix, val in enumerate(error_mean)])
        output = [y if x == 1 else (
            y+1) % 2 for x, y in zip(errors, self.df.loc[:, 'class'].astype(int))]
        #roc = roc_auc_score(self.df.loc[n_train+n_test:,'class'].astype(int).values, output)
        #f1 = f1_score(self.df.loc[n_train+n_test:,'class'].astype(int).values, output)
        somme = np.sum(errors)
        mean = np.mean(errors)
        #print("mean: %.3f, sum: %.3f, f1: %.3f, roc: %.3f"%(mean, somme, f1, roc))
        #comparison_df.loc['DDM-Concept_drift'] = [mean, somme, f1, roc]


if __name__ == '__main__':
    D_G = Drift_generators()
    D_G.smooth_concept_drift()
    df = D_G.get_circles
    print("done")
