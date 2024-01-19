import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import os
import pickle
import seaborn as sns

from scipy.io import arff


class Drift:
    def __init__(self, is_abrupt=True, start=None, end=None, characteristic=None):
        self.is_abrupt = is_abrupt
        if start != None:
            if start > 0 and start < 1:
                # TODO: add percentage of length of dataset support
                print("can do percentage assingnment")
            else:
                self.start = int(start)
        if end != None:
            if end > 0 and end < 1:
                print("can do percentage assingnment")
            else:
                self.end = int(end)
        self.characteristic = characteristic

    def __repr__(self):
        if self.is_abrupt == True:
            return f"Abrupt: at {self.start}, decision: {self.characteristic}"
        else:
            return f"Smooth: from {self.start}, to {self.end}, decision {self.characteristic}"


class Drift_generators:  # TODO: consider wether feat_random seed should be setable in drift generation fcts
    def __init__(
        self,
        n_samples=10**4,
        n_features=2,
        feature_random_seed=None,
        classif=True,
        model_random_seed=42,
    ):
        self.n = n_samples
        if feature_random_seed is None:
            feature_random_seed = np.random.randint(10000)
        self.random_seed = feature_random_seed
        self.model_random_seed = model_random_seed
        np.random.seed(seed=feature_random_seed)
        # if(n_features == 2):

        #    self.df = pd.DataFrame(columns =['x','y','class'])
        #    self.df['x'] = np.random.rand(n_samples)
        #    self.df['y'] = np.random.rand(n_samples)
        # else:
        self.df = pd.DataFrame(columns=["class"])
        for feat in range(n_features):
            self.df[feat] = np.random.rand(n_samples)
        self.n_features = n_features

        self.default_decision = [0.4 for i in range(self.n_features)]
        # apply default decision
        # hypersphere centered on 0.7,0.7,...,0.7 with default radius of 0.25
        self.classif = classif
        self.n_train = min(1000, int(n_samples / 10))
        self.n_test = int(0.2 * self.n_train)
        self.drift_points = []
        self.drifts = []
        self.drift_name = "NONE"
        self.model = None
        self.trained = False
        if self.classif:
            self.objective_col = "class"
        else:
            self.objective_col = "ts"
        self.noise_rate = 0

    # def print_df_names(self):
    def load_dataset_df(self, df, classif=True, drifts=None, df_name=None):
        self.df = df
        if df_name is not None:
            self.drift_name = df_name
        if classif:
            self.n_features = len(df.columns.drop("class"))
            self.objective_col = "class"
        else:  # TODO: consider changing ts to another more clear name.
            self.classif = False
            self.n_features = len(df.columns.drop("ts"))
            self.objective_col = "ts"
        self.n = len(df)
        self.n_train = min(1000, int(self.n / 10))
        self.n_test = int(0.2 * self.n_train)
        self.default_decision = None
        if drifts is not None:
            self.drifts = drifts
            self.drift_points = [d.start for d in self.drifts]

    def load_df(self, df_name, n_train=None, n_test=None, nrows=60000):
        # TODO: test skmultiflow.data.ConceptDriftStream¶
        data_path = os.environ.get("DATA_PATH")  # TODO: put as a env var
        if(data_path[-1]!="/"):
            #data_path[-1] += "/"
            df = pd.read_csv(data_path + "/" + df_name + ".csv", nrows=nrows)
        else:
            df = pd.read_csv(data_path + df_name + ".csv", nrows=nrows)
        self.n = len(df)
        self.drift_name = df_name
        if "class" in df.columns:
            if not df.columns[-1] == "class":
                df.columns = np.concatenate(
                    [df.drop(columns="class").columns, ["class"]]
                )
            df.columns = [
                "class" if x == "class" else i for i, x in enumerate(df.columns)
            ]

        self.df = df
        self.n_features = len(df.columns) - 1
        if n_train != None:
            self.n_train = n_train
        else:
            self.n_train = min(1000, int(self.n / 10))
        if n_test != None:
            self.n_test = n_test
        else:
            self.n_test = int(0.2 * self.n_train)

        # n_seen = self.n_train + self.n_test
        # train loaded DF
        # X, y = self.df.loc[:n_seen].drop(
        #    columns=['class']), self.df.loc[:n_seen, 'class']
        X_train, y_train = self.get_x_y(ind_end=self.n_train)
        # X_train, y_train = X.loc[:self.n_train], y.loc[:self.n_train]
        # X_test, y_test = X.loc[self.n_train:], y.loc[self.n_test:]
        # train initial model
        model = LGBMClassifier(random_state=self.model_random_seed, verbose=-1)
        model.fit(X_train, y_train)
        # self.df = self.df.loc[n_seen:]
        self.model = model
        self.trained = True
        # check if we set drift points
        if df_name in [
            "sine1",
            "sine2",
            "sine1_short",
            "sine2_short",
            "stagger",
            "stagger_short",
        ]:
            if "short" in df_name:
                self.drifts = [
                    Drift(
                        is_abrupt=True,
                        start=int(1 / 3 * self.n),
                        characteristic=df_name.split("_")[0],
                    ),
                    Drift(
                        is_abrupt=True,
                        start=int(2 / 3 * self.n),
                        characteristic=df_name.split("_")[0],
                    ),
                ]
                self.drift_points = [d.start for d in self.drifts]
            elif "sine" in df_name:
                self.drifts = [
                    Drift(is_abrupt=True, start=20020, characteristic="sine"),
                    Drift(is_abrupt=True, start=40030, characteristic="sine"),
                ]
                self.drift_points = [d.start for d in self.drifts]
            elif "stagger" in df_name:
                self.drifts = [
                    Drift(is_abrupt=True, start=33400, characteristic="stagger"),
                    Drift(is_abrupt=True, start=66700, characteristic="stagger"),
                ]
                # TODO: create a func drift points that does calculate the drift points at each call to avoid duplicate and conflicting values with self.drift
                self.drift_points = [d.start for d in self.drifts]

    #    def set_drift_centers

    def set_default_decision(self, beg=None, end=None, decision=None):
        if beg == None:
            beg = 0
        if end == None:
            end = self.n
        if decision != None:
            self.default_decision = decision
        self.df.loc[beg:end, "class"] = (
            self.df.loc[beg:end]
            .drop(columns=["class"])
            .apply(
                lambda data: self.is_in_hypersphere(data.values, self.default_decision),
                axis=1,
            )
        )

    def set_decision(self, beg, end, decision, verbose=False):
        if verbose:
            print(f"Setting {decision} from {beg} to {end}")
        self.df.loc[beg:end, "class"] = (
            self.df.loc[beg:end]
            .drop(columns=["class"])
            .apply(lambda data: self.is_in_hypersphere(data.values, decision), axis=1)
        )

    def set_class(self, beg, end, dec_func, verbose=False):
        if verbose:
            print(f"Setting {dec_func} from {beg} to {end}")
        self.df.loc[beg:end, "class"] = (
            self.df.loc[beg:end]
            .drop(columns=["class"])
            .apply(lambda data: dec_func(data.values), axis=1)
        )

    def is_in_hypersphere(self, point, center, r=0.5):
        V = 0.4
        if self.n_features == 2:
            r = V ** (0.5) / np.pi ** (0.5)
        elif self.n_features == 3:
            r = ((3 * V) / (np.pi * 4)) ** (1 / 3)
        elif self.n_features == 4:
            r = ((2 * V) ** (1 / 4)) / (np.pi ** (1 / 2))
        else:
            r = 0.4
        return sum([(p - c) ** 2 for p, c in zip(point, center)]) < r**2

    def abrupt_concept_drift(
        self, n_drift=1, drifts=None, drift_points=None, default_decision=None
    ):
        if drifts is None:
            if drift_points is None:
                drift_points = [
                    (i_drift + 1) * self.n / (1 + n_drift) for i_drift in range(n_drift)
                ]
            for i_drift, d_p in enumerate(drift_points):
                drift = Drift(
                    is_abrupt=True,
                    start=d_p,
                    characteristic=[
                        0.7 - 0.3 * (i_drift % 2) for i in range(self.n_features)
                    ],
                )
                self.drifts.append(drift)
        else:
            self.drifts += drifts

        self.drifts = sorted(self.drifts, key=lambda x: x.start)

        if default_decision != None:
            self.default_decision = default_decision
        self.set_decision(
            beg=0, end=self.drifts[0].start, decision=self.default_decision
        )

        for i, d in enumerate(self.drifts):
            if i < len(self.drifts) - 1:
                self.set_decision(
                    beg=d.start, end=self.drifts[i + 1].start, decision=d.characteristic
                )
            else:
                self.set_decision(beg=d.start, end=self.n, decision=d.characteristic)

        self.drift_points = [d.start for d in self.drifts]
        self.drift_name = "abrupt concept drift"
        self.circle_centers = [d.characteristic for d in self.drifts]

    def brutal_concept_drift(self, n_drift=1, circle_centers=None, drift_points=None):
        # TODO: check why there are less positive instances before drift than after drift even if the drift point is centered
        """Simulate concept drift by arbitrarily changing the decision boundary"""
        if circle_centers != None:
            if len(circle_centers) == n_drift:
                circle_centers = [self.default_decision] + circle_centers
            elif len(circle_centers) != n_drift + 1:  # ELSE default case
                circle_centers = [
                    [0.7 for i in range(self.n_features)] for d in range(n_drift)
                ]
        else:
            circle_centers = [
                [0.7 for i in range(self.n_features)] for d in range(n_drift)
            ]  # drift circles

        if (
            drift_points == None
        ):  # if no drift_point is specified just regularly drift along the dataset
            drift_points = [
                int(d * (self.n - self.n_train) / (n_drift + 1) + self.n_train)
                for d in range(1, n_drift + 2)
            ]
        else:
            if len(drift_points) == n_drift:
                drift_points.append(self.n)
            elif len(drift_points) < n_drift:
                drift_points = drift_points + [
                    int(
                        d
                        * (self.n - drift_points[-1])
                        / (n_drift + 1 - len(drift_points))
                        + drift_points[-1]
                    )
                    for d in range(len(drift_points), n_drift + 1)
                ]
            else:
                drift_points = [
                    int(d * (self.n - self.n_train) / (n_drift + 1) + self.n_train)
                    for d in range(1, n_drift + 2)
                ]

        self.set_default_decision(0, drift_points[0], decision=circle_centers[0])
        for d in range(n_drift):
            self.df.loc[drift_points[d] : drift_points[d + 1], "class"] = (
                self.df[drift_points[d] : drift_points[d + 1]]
                .drop(columns=["class"])
                .apply(
                    lambda data: self.is_in_hypersphere(
                        data.values, circle_centers[d + 1]
                    ),
                    axis=1,
                )
            )

        self.drift_points = drift_points[:-1]
        self.drift_name = "brutal concept drift"
        self.circle_centers = circle_centers

    def smooth_concept_drift(self, n_drift=1, drifts=None, default_decision=None):
        """Simulate smooth concept drift by arbitrarily changing the decision boundary"""
        if drifts is None:
            for i_drift in range(n_drift):
                drift = Drift(
                    is_abrupt=False,
                    start=(i_drift + 1) * self.n / (2 + n_drift),
                    end=(i_drift + 2) * self.n / (2 + n_drift),
                    characteristic=[
                        0.7 - 0.3 * (i_drift % 2) for i in range(self.n_features)
                    ],
                )
                self.drifts.append(drift)
        else:
            self.drifts += drifts

        if default_decision != None:
            self.default_decision = default_decision
        self.set_decision(
            beg=0, end=self.drifts[0].start, decision=self.default_decision
        )
        current_decision = self.default_decision

        for d in self.drifts:
            if d.characteristic != current_decision:
                s_x = [
                    (d.characteristic[f] - current_decision[f]) / (d.end - d.start)
                    for f in range(self.n_features)
                ]
                for i, k in enumerate(range(d.start, d.end)):
                    point = self.df.drop(columns=["class"]).loc[k].values
                    center = [
                        (current_decision[f] + s_x[f] * i)
                        for f in range(self.n_features)
                    ]
                    self.df.loc[k, "class"] = self.is_in_hypersphere(point, center)
                current_decision = d.characteristic
            else:
                self.set_decision(d.start, d.end, current_decision)
        self.set_decision(beg=d.end, end=self.n, decision=current_decision)

        self.drift_points = sorted(
            [d.start for d in self.drifts] + [d.end for d in self.drifts]
        )
        self.drift_name = "smooth concept drift"
        self.circle_centers = [d.characteristic for d in self.drifts]

    def gradual_concept_drift(
            self, n_drift=1, drifts=None, default_decision=None, n_grad=5
        ):

        """Simulate smooth concept drift by arbitrarily changing the decision boundary"""
        if drifts is None:
            for i_drift in range(n_drift):
                drift = Drift(
                    is_abrupt=False,
                    start=(i_drift + 1) * self.n / (2 + n_drift),
                    end=(i_drift + 2) * self.n / (2 + n_drift),
                    characteristic=[
                        0.7 - 0.3 * (i_drift % 2) for i in range(self.n_features)
                    ],
                )
                self.drifts.append(drift)
        else:
            self.drifts += drifts

        if default_decision != None:
            self.default_decision = default_decision
        self.set_decision(
            beg=0, end=self.drifts[0].start, decision=self.default_decision
        )
        current_decision = self.default_decision
        
        for d in self.drifts:
            if d.characteristic != current_decision:
                step_size = int((d.end - d.start) / n_grad)
                for i, beg_seq in enumerate(range(d.start, d.end, step_size)):
                    frac = int((i+1)*step_size/(n_grad+1))
                    self.set_decision(
                            beg=beg_seq, end=beg_seq+frac, decision=d.characteristic
                        )
                    self.set_decision(
                            beg=beg_seq+frac, end=beg_seq+step_size, decision=current_decision
                        )
                    
                self.set_decision(beg_seq+step_size, d.end, decision = current_decision)
            else:
                self.set_decision(beg=d.start, end=d.end, decision=current_decision)
                
            current_decision = d.characteristic
        self.set_decision(beg=d.end, end=self.n, decision=current_decision)

        self.drift_points = sorted(
            [d.start for d in self.drifts] + [d.end for d in self.drifts]
        )
        self.drift_name = "gradual concept drift"
        self.circle_centers = [d.characteristic for d in self.drifts]

    def new_smooth_concept_drift(
        self, n_drift=1, drift_points=None, circle_centers=None
    ):
        # TODO check how drift points are set.
        # TODO: Check if we can delete this
        """
        Smoothly change the decision frontier in bettween the drift points from one circle center to another
        """

        # if no drift_point is specified just regularly drift along the whole dataset
        if drift_points == None:
            drift_points = [
                int(d * (self.n - self.n_train) / (n_drift + 1) + self.n_train)
                for d in range(1, n_drift + 2)
            ]
        else:
            if len(drift_points) == n_drift:
                drift_points.append(self.n)
            elif len(drift_points) < n_drift:
                drift_points = drift_points + [
                    int(
                        d
                        * (self.n - drift_points[-1])
                        / (n_drift + 1 - len(drift_points))
                        + drift_points[-1]
                    )
                    for d in range(len(drift_points), n_drift + 1)
                ]

        if circle_centers == None:  # randomly fix some circle_centers
            # circle_centers = [[0.3/10 for i in range(self.n_features)] for d in range(n_drift)]#drift circles
            circle_centers = [
                [0.5 + (np.random.randint(3) - 1) / 8 for i in range(self.n_features)]
                for d in range(n_drift)
            ]  # drift circles
        elif len(circle_centers) == n_drift:
            # print("A")
            circle_centers = [self.default_decision] + circle_centers
        elif len(circle_centers) > n_drift + 1:
            circle_centers = circle_centers[: n_drift + 1]
        elif len(circle_centers) < n_drift:
            print("problem")
            # circle_centers = circle_centers[:n_drift+1]

        # print(drift_points,circle_centers)

        # set decision before drift
        self.set_default_decision(0, drift_points[0])
        # set decision during drift
        for d in range(n_drift):
            if circle_centers[d + 1] != circle_centers[d]:
                s_x = [
                    (circle_centers[d + 1][i] - circle_centers[d][i])
                    / (drift_points[d + 1] - drift_points[d])
                    for i in range(self.n_features)
                ]
                for i, k in enumerate(range(drift_points[d], drift_points[d + 1])):
                    point = self.df.drop(columns=["class"]).loc[k].values
                    center = [
                        (circle_centers[d][f] + s_x[f] * i)
                        for f in range(self.n_features)
                    ]
                    self.df.loc[k, "class"] = self.is_in_hypersphere(point, center)
            else:
                self.df.loc[drift_points[d] : drift_points[d + 1], "class"] = (
                    self.df.loc[drift_points[d] : drift_points[d + 1]]
                    .drop(columns=["class"])
                    .apply(
                        lambda data: self.is_in_hypersphere(
                            data.values, circle_centers[d + 1]
                        ),
                        axis=1,
                    )
                )
        # set decision after drift as ended
        if drift_points[-1] != self.n:
            self.df.loc[drift_points[-1] :, "class"] = (
                self.df.loc[drift_points[-1] :]
                .drop(columns=["class"])
                .apply(
                    lambda data: self.is_in_hypersphere(
                        data.values, circle_centers[-1]
                    ),
                    axis=1,
                )
            )

        self.drift_points = drift_points  # [:-1]
        self.drift_name = "smooth concept drift"
        self.circle_centers = circle_centers

    def abrupt_covariate_drift(
        self, d_centers=[("x", 0, 0.2, 0.8, 0.9), ("y", 0.5, 0.7, 0, 0.5)]
    ):
        for feat, beg, end, min_f, max_f in d_centers:
            beg, end = int(self.n * beg), int(self.n * end)
            self.df.loc[beg : end - 1, feat] = (
                np.random.rand(end - beg) * (max_f - min_f) + min_f
            )

            drift = Drift(
                is_abrupt=False, start=beg, end=end, characteristic=[feat, min_f, max_f]
            )
            if beg != 0:
                self.drifts.append(drift)
        self.set_default_decision()

        self.drift_points = sorted(
            [d.start for d in self.drifts] + [d.end for d in self.drifts]
        )
        self.drift_name = "abrupt covariate"
        self.circle_centers = None

    def brutal_covariate_drift(
        self, d_centers=[("x", 0, 0.2, 0.8, 0.9), ("y", 0.5, 0.7, 0, 0.5)]
    ):
        # TODO: Deprecated, check if we can delete
        """
        The drift characteristics are defined the following way
        feature_drifting, start_point(as a % of len(df)), end_point(as a % of len(df)),
        min_feature_range(as a % of feat_values), max_feature range(as a % of feat_values)

        first make the feature drift
        then specify the decision boundary
        """

        drift_points = [
            int(x * self.n)
            for x in np.unique(np.ravel([(x[1], x[2]) for x in d_centers]))
        ]
        # self.set_default_decision()

        for feat, beg, end, min_f, max_f in d_centers:
            beg, end = int(self.n * beg), int(self.n * end)
            self.df.loc[beg : end - 1, feat] = (
                np.random.rand(end - beg) * (max_f - min_f) + min_f
            )

        self.set_default_decision()

        self.drift_points = drift_points
        self.drift_name = "abrupt covariate"
        self.circle_centers = None

    def cyclic_abrupt_concept_drift(self, n_drift):
        circle_centers = [
            (0.5 * np.cos(x), 0.5 * np.sin(x))
            for x in np.linspace(0, 2 * np.pi, n_drift + 1)
        ]
        self.brutal_concept_drift(n_drift=n_drift, circle_centers=circle_centers)
        self.drift_name = "cyclic concept"

    def back_and_forth_abrupt_drift(self, circle_centers=None, drift_points=None):
        if circle_centers == None:
            circle_centers = [
                [0.7 for i in range(self.n_features)],
                self.default_decision,
            ]
        self.brutal_concept_drift(
            n_drift=2, circle_centers=circle_centers, drift_points=drift_points
        )
        self.drift_name = "back and forth abrupt"

    def back_and_forth_smooth_drift(self):
        circle_centers = [
            [0.7 for i in range(self.n_features)],
            self.default_decision,
            self.default_decision,
        ]
        self.new_smooth_concept_drift(n_drift=3, circle_centers=circle_centers)
        self.drift_name = "back and forth smooth"

    def add_noise(self, noise_rate=0.001):
        # select and corrupt some rows labels
        # noisy_rows = self.df.sample(int(self.n*noise_rate))
        if noise_rate > 0:
            X, y = self.get_x_y(ind_start=0, ind_end=self.n)
            noisy_inds = y.sample(int(self.n * noise_rate)).index
            # Strange bug with covar if we do not cast....
            self.noise_rate = noise_rate
            self.df.loc[noisy_inds, self.objective_col] = list(
                y[noisy_inds]
                .apply(lambda x: np.random.choice([a for a in y.unique() if a != x]))
                .values
            )
            self.drift_name += f" noisy{str(noise_rate)[2:]}"

    def add_class_conditional_noise(self, noise_rate=0.001, noisy_class=0):
        # select and corrupt some rows labels
        # noisy_rows = self.df.sample(int(self.n*noise_rate))
        if noise_rate > 0:
            X, y = self.get_x_y(ind_start=0, ind_end=self.n)
            noisy_inds = y[y == noisy_class].sample(int(self.n * noise_rate)).index
            # Strange bug with covar if we do not cast....
            self.noise_rate = noise_rate
            self.df.loc[noisy_inds, self.objective_col] = list(
                y[noisy_inds]
                .apply(lambda x: np.random.choice([a for a in y.unique() if a != x]))
                .values
            )
            self.drift_name += f" noisy{str(noise_rate)[2:]}"

    # def prior_proba_drift():
    #    return

    def get_circles(self):
        return self.df

    def train(self, n_train=None, n_test=None, shuffle=False):
        """
        Beware when using the shuffle arg as it is contrary to the online learning setting order may have its importance.
        """
        if n_train != None:
            self.n_train = n_train
        if n_test != None:
            self.n_test = n_test
        X, y = self.get_x_y(ind_start=0, ind_end=self.n_train + self.n_test)

        X_train, y_train = X.iloc[: self.n_train], y.iloc[: self.n_train]
        X_test, y_test = X.iloc[self.n_train :], y.iloc[self.n_train :]
        if shuffle:
            X_train = X_train.sample(frac=1)
            y_train = y_train.loc[X_train.index]
        # train initial model
        if self.model is None:
            if self.classif:
                self.model = LGBMClassifier(
                    random_state=self.model_random_seed, verbose=-1
                )
            else:
                self.model = LGBMRegressor(random_state=self.model_random_seed)
        self.model.fit(X_train, y_train)
        self.score = self.model.score(X_test, y_test)

        self.trained = True

    #        print("model trained, score on test set: %.3f"%self.score)

    def retrain_model(self, shuffle=False, model_type="lgbm"):
        self.model_random_seed = np.random.randint(1000)
        if model_type == "lgbm":
            self.model = LGBMClassifier(
                random_state=self.model_random_seed, n_estimators=10, verbose=-1
            )
        elif model_type == "GaussianNB":
            self.model = GaussianNB()
        elif model_type == "DecisionTreeClassifier":
            self.model = DecisionTreeClassifier(random_state=self.model_random_seed)
        else:
            self.model = LGBMClassifier(
                random_state=self.model_random_seed, n_estimators=10, verbose=-1
            )
        self.trained = False
        # THODO TEST IF SHUFFLE DOESNT FUCK THINGS UP
        self.train(shuffle=shuffle)

    def get_error(self):
        """
        Compute all errors on the full dataset
        returns: errors, retrain_indices
        """
        if not self.trained:  # check whether a model was trained
            self.train()

        X, y = self.get_x_y()
        # X = self.df.loc[:].drop(columns=['class'])
        # y = self.df.loc[:, 'class'].astype("category")
        if self.classif:
            errors = [int(p == y) for p, y in zip(self.model.predict(X.values), y)]
        else:
            errors = [abs(p - y) for p, y in zip(self.model.predict(X.values), y)]

        self.errors = errors
        return (errors, [])

    def get_preds(self):
        """
        Compute all errors on the full dataset
        returns: errors, retrain_indices
        """
        if not self.trained:  # check whether a model was trained
            self.train()

        X, y = self.get_x_y()
        # X = self.df.loc[:].drop(columns=['class'])
        # y = self.df.loc[:, 'class'].astype("category")

        preds = self.model.predict(X.values)

        self.preds = preds
        return (preds, [])

    def n_seen(self):
        return self.n_train + self.n_test

    # ---------------------------------SINE---------------------------------

    def fct_drift(
        self,
        n_drift=1,
        drifts=None,
        fct=lambda x: x[1] - np.sin(x[0]),
        drift_name="Sine1",
    ):
        if drifts is None:
            for i_drift in range(n_drift):
                drift = Drift(
                    is_abrupt=True,
                    start=(i_drift + 1) * self.n / (1 + n_drift),
                    characteristic="Above" if i_drift % 2 == 0 else "Bellow",
                )
                self.drifts.append(drift)
        else:
            self.drifts += drifts
        self.drifts = sorted(self.drifts, key=lambda x: x.start)

        def above(x):
            return int(fct(x) > 0)

        def bellow(x):
            return int(fct(x) < 0)

        if self.drifts[0].characteristic == "Above":
            self.default_decision = "Bellow"
            self.set_class(
                beg=0, end=self.drifts[0].start, dec_func=bellow
            )  # , decision="Bellow")
        else:
            self.default_decision = "Above"
            self.set_class(
                beg=0, end=self.drifts[0].start, dec_func=above
            )  # , decision="Above")

        for i, d in enumerate(self.drifts):
            if i < len(self.drifts) - 1:
                beg, end = d.start, self.drifts[i + 1].start
            else:
                beg, end = d.start, self.n
            if d.characteristic == "Above":
                self.set_class(
                    beg=beg, end=end, dec_func=above
                )  # , decision=d.characteristic)
            else:
                self.set_class(
                    beg=beg, end=end, dec_func=bellow
                )  # , decision=d.characteristic)

        self.drift_points = [d.start for d in self.drifts]
        self.drift_name = drift_name

    def sine1_drift(
        self, n_drift=1, drifts=None
    ):  # TODO: add check to have only 2 feats
        self.fct_drift(
            n_drift=n_drift, fct=lambda x: x[1] - np.sin(x[0]), drift_name="Sine1"
        )

    def sine2_drift(self, n_drift=1, drifts=None):
        self.fct_drift(
            n_drift=n_drift,
            fct=lambda x: x[0] - 0.5 + 0.3 * np.sin(3 * np.pi * x[0]),
            drift_name="Sine2",
        )

    # def stagger_drift(self, n_drift=1, drifts=None):#TODO stagger drift generator ? ?

    # ---------------------------------LOAD---------------------------------

    def load_results(self, funcs, dataset="brutal_concept_backforth", n_load=10):
        """Previous name was load_data"""
        # TODO: Check function name and check if useful

        print([x[8:] for x in funcs])

        n_iter = min(
            n_load,
            min(
                [
                    len(
                        [
                            f
                            for f in os.listdir(path)
                            if os.path.isfile(os.path.join(path, f))
                        ]
                    )
                    for path in [
                        "/home/bastien/Documents/labs/data/results/"
                        + dataset
                        + "/"
                        + func[8:]
                        for func in funcs
                    ]
                ]
            ),
        )

        error_list = [[[] for i in range(len(funcs))] for k in range(n_iter)]
        detections = [[[] for i in range(len(funcs))] for k in range(n_iter)]

        for i, func in enumerate(funcs):
            path = (
                "/home/bastien/Documents/labs/data/results/" + dataset + "/" + func[8:]
            )
            files = np.sort(
                [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            )[-n_iter:]
            for k, f in enumerate(files):
                err, detec = pickle.load(open(path + "/" + f, "rb"))
                error_list[k][i] = err
                detections[k][i] = detec
        print("\n%d results loaded for each policy (%d funcs)" % (n_iter, len(funcs)))
        return (error_list, detections)

    # _---------------------------------PERF------------------#TODO: Check if is at correct place

    def compute_retrain_perf(self, errors):
        """
        Not checked yet
        """
        #    error_measure_corrected_by_variance =  np.asarray([math.sqrt(val * (1 - val) / (ix +1)) for
        #                                                         ix, val in enumerate(error_mean)])
        output = [
            y if x == 1 else (y + 1) % 2
            for x, y in zip(errors, self.df.loc[:, "class"].astype(int))
        ]
        # roc = roc_auc_score(self.df.loc[n_train+n_test:,'class'].astype(int).values, output)
        # f1 = f1_score(self.df.loc[n_train+n_test:,'class'].astype(int).values, output)
        somme = np.sum(errors)
        mean = np.mean(errors)
        # print("mean: %.3f, sum: %.3f, f1: %.3f, roc: %.3f"%(mean, somme, f1, roc))
        # comparison_df.loc['DDM-Concept_drift'] = [mean, somme, f1, roc]

    def plot_circles(self):
        if self.n_features == 2:
            cmap = matplotlib.cm.get_cmap("tab10")

            fix, ax = plt.subplots(nrows=1, ncols=1, figsize=[20, 10])
            # drift_centers):
            for i, (c_x, c_y) in enumerate(self.circle_centers):
                circle = plt.Circle(
                    (c_x, c_y),
                    0.25,
                    color=cmap(i),
                    fill=False,
                    label="drift_" + str(i + 1),
                )
                ax.add_patch(circle)

            c_x, c_y = self.default_decision
            # for (c_x, c_y) in self.default_decision:# drift_centers):
            circle = plt.Circle(
                (c_x, c_y), 0.25, color="green", fill=False, label="default decision"
            )
            ax.add_patch(circle)
            # can be replaced with a proper setting of plot frame
            ax.scatter(
                [x[0] for x in self.circle_centers],
                [x[1] for x in self.circle_centers],
                marker="x",
            )
            ax.legend()
            ax.set_title(self.drift_name + "\nmean error")
            plt.show()
        else:
            print("too many features to plot maybe PCA to implement later")

    def plot_drift(self, sample_frac=0.1):
        fig, ax = plt.subplots(
            nrows=self.df.loc[:, self.objective_col].nunique(),
            ncols=self.n_features,
            figsize=[
                self.n_features * 2.5,
                2.5 * self.df.loc[:, self.objective_col].nunique(),
            ],
        )

        feat_cols = [x for x in self.df.columns if x != self.objective_col]
        if len(self.drifts) == 0:
            self.drifts = [Drift(start=0)]
        for i, class_val in enumerate(self.df.loc[:, self.objective_col].unique()):
            ind_start = 0
            for i_d, d in enumerate(self.drifts):
                if i_d == (len(self.drifts) - 1):
                    ind_end = self.n
                else:
                    ind_end = self.drifts[i_d + 1].start
                bef_X, bef_y = self.get_x_y(
                    ind_start=ind_start, ind_end=d.start, sample=sample_frac
                )
                bef_X = bef_X[bef_y == class_val]
                #                bef = self.df[self.df.loc[:, self.objective_col] ==
                #                              class_val].loc[ind_start: d.start, feat_cols].sample(frac=sample_frac).sort_index()
                if not d.is_abrupt:
                    dur_X, dur_y = self.get_x_y(
                        ind_start=d.start, ind_end=d.end, sample=sample_frac
                    )
                    dur_X = dur_X[dur_y == class_val]
                    aft_X, aft_y = self.get_x_y(
                        ind_start=d.end, ind_end=ind_end, sample=sample_frac
                    )
                    aft_X = aft_X[aft_y == class_val]
                    for i_feat, feat in enumerate(bef_X.columns):
                        if len(bef_X) > 0:
                            sns.regplot(
                                x=np.linspace(ind_start, d.start, len(bef_X)),
                                y=bef_X.loc[:, feat],
                                label="Before",
                                ax=ax[i, i_feat],
                            )
                        if len(dur_X) > 0:
                            sns.regplot(
                                x=np.linspace(d.start, d.end, len(dur_X)),
                                y=dur_X.loc[:, feat],
                                label="During",
                                ax=ax[i, i_feat],
                            )
                        if len(aft_X) > 0:
                            sns.regplot(
                                x=np.linspace(d.end, ind_end, len(aft_X)),
                                y=aft_X.loc[:, feat],
                                label="After",
                                ax=ax[i, i_feat],
                            )
                    ind_start = d.end
                else:
                    aft_X, aft_y = self.get_x_y(
                        ind_start=d.start, ind_end=ind_end, sample=sample_frac
                    )
                    aft_X = aft_X[aft_y == class_val]
                    for i_feat, feat in enumerate(feat_cols):
                        if len(bef_X) > 0:
                            sns.regplot(
                                x=np.linspace(ind_start, d.start, len(bef_X)),
                                y=bef_X.loc[:, feat],
                                label="Before",
                                ax=ax[i, i_feat],
                            )
                        if len(aft_X) > 0:
                            sns.regplot(
                                x=np.linspace(d.start, ind_end, len(aft_X)),
                                y=aft_X.loc[:, feat],
                                label="After",
                                ax=ax[i, i_feat],
                            )
                    ind_start = d.start
            for i_feat, feat in enumerate(feat_cols):
                # ax[i, feat].legend()
                ax[i, i_feat].set_title(f"col : {feat}")
                ax[i, i_feat].set_xlabel("positive class instances")
                ax[i, i_feat].set_ylabel(f"feat:{feat} value")
        return fig

    def get_x_y(self, ind_start=None, ind_end=None, sample=None):
        if ind_start is None:
            ind_start = 0
        if ind_end is None:
            ind_end = len(self.df)
        if self.classif:
            X = self.df.iloc[ind_start:ind_end].drop(columns=[self.objective_col])
            y = self.df.iloc[ind_start:ind_end][self.objective_col].astype("category")
        else:
            X = self.df.iloc[ind_start:ind_end].drop(columns=[self.objective_col])
            y = self.df.iloc[ind_start:ind_end][self.objective_col]
        if sample is not None:
            X = X.sample(frac=sample).sort_index()
            y = y.loc[X.index]

        return (X, y)


def load_river_data(dataset_name="Elec2", n_samples=None, drop_dtypes=False):
    if dataset_name == "insects":
        exec(f"from river.datasets.insects import Insects as riverDataset", globals())
    elif dataset_name == "taxis":
        exec(f"from river.datasets.taxis import Taxis as riverDataset", globals())
    else:
        exec(f"from river.datasets import {dataset_name} as riverDataset", globals())
    data = riverDataset()
    if "download" in dir(data):
        data.download()
    df = pd.DataFrame(columns=list([x for x in data.take(1)][0][0].keys()))
    label = []
    if n_samples is None:
        n_samples = data.n_samples
    df = pd.DataFrame([x[0] for x in data.take(n_samples)])
    df.loc[:, "class"] = [x[1] for x in data.take(n_samples)]
    if drop_dtypes:
        # TODO: add support for categorical and datetime data
        df = df.drop(
            columns=[
                x
                for x in df.select_dtypes(include=[np.datetime64, np.object_]).columns
                if x != "class"
            ]
        )
    return df


def create_df_from_ts(
    ts,
    n_rolls=4,
    roll_width_step=1,
    n_lags=4,
    lag_Step=1,
    ewm_alphas=[0.99, 0.95, 0.7, 0.1],
):
    df = pd.DataFrame(ts, columns=["ts"])

    # Create roll
    roll_widths = [int(roll_width_step * (n + 1)) for n in range(n_rolls)]
    for n, w in enumerate(roll_widths):
        df[f"roll{n}"] = df["ts"].shift(1).rolling(window=w).mean()

    lags = [int(lag_Step * n) for n in range(n_lags)]
    # Create lag
    for n, lag in enumerate(lags):
        df[f"lag{n}"] = df["ts"].shift(lag)

    ewm_alphas = [0.99, 0.95, 0.7, 0.1]
    # Create lag
    for a in ewm_alphas:
        df[f"ewm{a}"] = df["ts"].shift(1).ewm(alpha=a).mean()
    # dfz
    df = df.dropna()

    return df


def get_insect_files():
    dict_insect_filenames = {
        "INSECTS-abrupt_imbalanced_norm.arff": "Abrupt (imbal.)",
        "INSECTS-abrupt_balanced_norm.arff": "Abrupt (bal.)",
        "INSECTS-gradual_balanced_norm.arff": "Incremental-gradual (bal.)",
        "INSECTS-gradual_imbalanced_norm.arff": "Incremental-gradual (imbal.)",
        "INSECTS-incremental-abrupt_balanced_norm.arff": "Incremental-abrupt-reoccurring (bal.)",
        "INSECTS-incremental-abrupt_imbalanced_norm.arff": "Incremental-abrupt-reoccurring (imbal.)",
        "INSECTS-incremental-reoccurring_balanced_norm.arff": "Incremental-reoccurring (bal.)",
        "INSECTS-incremental-reoccurring_imbalanced_norm.arff": "Incremental-reoccurring (imbal.)",
    }
    return dict_insect_filenames


def get_dict_insects():
    dict_insects = {
        "Abrupt (imbal.)": [83859, 128651, 182320, 242883, 268380],
        "Abrupt (bal.)": [14352, 19500, 33240, 38682, 39510],
        "Incremental-gradual (bal.)": [14028],
        "Incremental-gradual (imbal.)": [58159],
        "Incremental-abrupt-reoccurring (bal.)": [26568, 53364],
        "Incremental-abrupt-reoccurring (imbal.)": [150683, 301365],
        "Incremental-reoccurring (bal.)": [26568, 53364],
        "Incremental-reoccurring (imbal.)": [150683, 301365],
    }
    return dict_insects


def load_insect_data(file_name, sample_rate=0.2):
    dict_insects = get_dict_insects()
    dict_insec_filenames = get_insect_files()

    data = arff.loadarff("../data/report_drift/" + file_name)
    df = pd.DataFrame(data[0]).sample(frac=sample_rate)
    df.loc[:, "class"] = df.loc[:, "class"].apply(lambda x: str(x)[2:-1])
    df.loc[:, "class"] = df.loc[:, "class"].astype("category")
    changepoints = dict_insects[dict_insec_filenames[file_name]]
    changepoints = [int(x * sample_rate) for x in changepoints]
    return (df, changepoints)
