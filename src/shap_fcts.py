import shap
import numpy as np


def init_shap(signal, Rt, bgd_type=None, n_samp_bgd=50, loss=None, objective_col="class"):
    """
    The background size is limited to be at most the size of train_test set.
    """
    if(bgd_type is not None):
        if(bgd_type == "worse"):
            X_bgrd = Rt.df.iloc[[x[0] for x in sorted(
                [(i, x) for i, x in enumerate(loss)], key=lambda x: x[1], reverse=True)][:n_samp_bgd]]
        elif(bgd_type == "best"):
            X_bgrd = Rt.df.iloc[[x[0] for x in sorted(
                [(i, x) for i, x in enumerate(loss)], key=lambda x: x[1], reverse=True)][-n_samp_bgd:]]
        elif(bgd_type == "sample"):
            X_bgrd = Rt.df.loc[:Rt.n_train+Rt.n_test].sample(min(n_samp_bgd, Rt.n_train+Rt.n_test))
        elif(bgd_type == "train"):
            X_bgrd = Rt.df.loc[:Rt.n_train +
                               Rt.n_test]
        elif(bgd_type == "small"):
            X_bgrd = Rt.df.loc[:Rt.n_train +
                                 Rt.n_test].sample(2)
        else:
            print(f"BGD Type \"{bgd_type}\" is not implemented")
        X_bgrd = X_bgrd.drop(columns=[objective_col])
        #print(X_bgrd)
    else:
        X_bgrd = Rt.df.loc[:Rt.n_train +
                            Rt.n_test].drop(columns=[objective_col])


    #if ("shap" == signal):
    explainer = shap.TreeExplainer(Rt.model, data=X_bgrd,
                                    feature_perturbation="interventional", model_output='log_loss')
    return(explainer)


def compute_shap_val(signal, Rt, explainer, x, label):
    #if("loss" in signal or "proba" in signal):
    try:
        point_shap_val = explainer.shap_values(X=np.array(x), y=[label])
    except:
        #print(np.array(x),label, explainer)
        #print(explainer.model)
        #
        # explainer.tree_limit = 100
        print("fail shav_val returning expected_value")
        #print(explainer.model)
        return(Rt.model.predict_proba(explainer.data).mean(0)[label])
    #else:
    #    point_shap_val = explainer.shap_values(
    #        x)[0]
    return(point_shap_val)
