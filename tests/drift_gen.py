import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.data_fcts import Drift_generators



#circle_centers = [[0.7 for i in range(self.n_features)], self.default_decision, self.default_decision]
D_G = Drift_generators(n_samples = 12000, n_features = 3)
D_G.new_smooth_concept_drift(n_drift=1, drift_points=[5000,9000], 
        circle_centers=[[0.5 for i in range(D_G.n_features)],[0.7 for i in range(D_G.n_features)]])# = None, circle_centers = None):
D_G = Drift_generators()
#print(D_G)

#D_G.df.loc[5000:, 'class'] =\
#        D_G.df.loc[5000:].drop(columns=['class']).apply(lambda data: D_G.is_in_hypersphere(data.values, [0.625, 0.5, 0.5]), axis=1)
