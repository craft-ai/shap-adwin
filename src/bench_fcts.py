import os
import pandas as pd
from ast import literal_eval
import numpy as np
from .retrain_fcts import Retrainer
import matplotlib.pyplot as plt

results_root_path = os.environ.get("RESULTS_ROOT_PATH")
                     

def bench_hyperparams_detector(
    retrain_fct_name,
    param_name_str,
    parameter_values,
    all_functions,
    all_noise_rates,
    save=True,
    path=None,
):
    noise_res = {}
    all_res = []
    for functions, noise_rate in zip(all_functions, all_noise_rates):
        alpha_res = {}

        for param in parameter_values:
            print(param_name_str, param)
            raw_res = []
            res_sucess = []
            res_perf = []
            names = []

            for f in functions:
                D_G = f()
                Rt = Retrainer(D_G)
                getattr(Rt, retrain_fct_name)(
                    **{param_name_str: param, "stop_first_detect": True}
                )

                a = D_G.drift_points[0]
                if len(Rt.detection_indices) > 0:
                    b = Rt.detection_indices[0]
                else:
                    b = -1
                if "nodrift" in D_G.drift_name:
                    success = b == -1
                    perf = -1
                else:
                    success = b > 0 and a < b
                    if success:
                        perf = b - a
                    else:
                        perf = D_G.n - D_G.drift_points[0]

                all_res.append(
                    [
                        noise_rate,
                        param,
                        success,
                        perf,
                        D_G.drift_name.replace(" ", "_"),
                    ]
                )
                # res_string = f"{}"# - Drift_point {a:.0f} - {b:.0f}"
                raw_res.append(b)
                res_sucess.append(success)
                res_perf.append(perf)
                names.append(D_G.drift_name)

            # print(res_sucess)

            res_df = pd.DataFrame(np.array([names, res_sucess, res_perf, raw_res]))
            alpha_res[param] = res_df

        noise_res[noise_rate] = alpha_res

    df_perf_detector = pd.DataFrame(
        all_res,
        columns=["noise_rate", param_name_str, "success", "performance", "drift_name"],
    )
    if save:
        if path is not None:
            df_perf_detector.to_csv(path)
        else:
            df_perf_detector.to_csv(
                os.environ.get("RESULTS_ROOT_PATH")
                + "/detectors_param/"
                + retrain_fct_name
                + "_"
                + param_name_str
                + ".csv"
            )

    return df_perf_detector


def eval_DG_model_save(
    generate_new_df, funcs, params_retrain, expe_set_name, n_iter, save=True, path=None
):
    results_df_model, results_df = None, None

    D_G = generate_new_df()
    results_df = eval_n_time_D_G(
        generate_new_df, funcs, params_retrain=params_retrain, n_iter=n_iter
    )
    if save:
        if path is None:
            results_path = (
                results_root_path
                + D_G.drift_name.replace(" ", "_")
                + f"_{len(D_G.df)}_df_reset_{expe_set_name}.csv"
            )
            results_df.to_csv(results_path)
        else:
            results_path = (
                path
                + D_G.drift_name.replace(" ", "_")
                + f"_{len(D_G.df)}_df_reset_{expe_set_name}.csv"
            )
            results_df.to_csv(results_path)
    # results_df_model = eval_n_time_model(
    #    D_G, funcs, params_retrain=params_retrain, n_iter=n_iter, shuffle=True)
    # if (save):
    #    results_path = results_root_path + \
    #        D_G.drift_name.replace(
    #            " ", "_")+f"_{len(D_G.df)}_model_reset_{expe_set_name}.csv"
    #    results_df_model.to_csv(results_path)
    return (results_df_model, results_df)


def eval_n_time_D_G(generate_df_func, funcs, params_retrain=None, n_iter=5):
    results_df = pd.DataFrame(columns=funcs)

    for i_d in range(n_iter):
        D_G = generate_df_func()
        results_D_G = []
        for fct in funcs:
            Rt = Retrainer(D_G)
            if params_retrain[fct] is not None:
                getattr(Retrainer, fct)(Rt, **params_retrain[fct])
            else:
                getattr(Retrainer, fct)(Rt)
            results_D_G.append(Rt.detection_indices)
        results_df = pd.concat(
            (
                results_df,
                pd.DataFrame(
                    data=[results_D_G], index=[f"{i_d}_{D_G.drift_name}"], columns=funcs
                ),
            )
        )

    return results_df


def eval_n_time_model(D_G, funcs, params_retrain=None, n_iter=5, shuffle=False):
    results_df = pd.DataFrame(columns=funcs)

    for i_d in range(n_iter):
        D_G.retrain_model(shuffle=shuffle)
        Rt = Retrainer(D_G)
        results_D_G = []
        for fct in funcs:
            if params_retrain[fct] is not None:
                getattr(Retrainer, fct)(Rt, **params_retrain[fct])
            else:
                getattr(Retrainer, fct)(Rt)
            results_D_G.append(Rt.detection_indices)
        results_df = pd.concat(
            (
                results_df,
                pd.DataFrame(
                    data=[results_D_G], index=[f"{i_d}_{D_G.drift_name}"], columns=funcs
                ),
            )
        )

    return results_df


def get_df_results(filename, path=None):
    # TODO: set as env var
    if path == None:
        results_path = os.environ.get("RESULTS_ROOT_PATH")
    else:
        results_path = path
    df_results = pd.read_csv(results_path + filename, index_col=0)
    for col in df_results.columns:
        df_results.loc[:, col] = df_results.loc[:, col].apply(literal_eval)
    return df_results


def get_df_metrics(results_df, drift_start=None):
    if drift_start is None:
        drift_start = 0
    # print("Drift start not set - considered as 0")
    # TP = (1-no_d)-f_d
    # FP = F_D
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    df_metrics = pd.DataFrame(columns=results_df.columns)
    # print(df_metrics)
    first_detecs = results_df.apply(
        lambda x: [a[0] if len(a) > 0 else np.nan for a in x], axis=0
    )
    no_detecs = results_df.apply(lambda x: [0 if len(a) > 0 else 1 for a in x]).mean()
    # print(no_detecs)
    df_metrics.loc["no"] = no_detecs

    false_detecs = first_detecs.apply(
        lambda x: [1 if a <= drift_start else 0 for a in x]
    ).mean()
    df_metrics.loc["false"] = false_detecs

    true_detecs = first_detecs.apply(
        lambda x: [1 if a > drift_start else 0 for a in x]
    ).mean()
    df_metrics.loc["TP"] = true_detecs

    df_metrics.loc["prec"] = df_metrics.apply(lambda x: x.TP / (x.TP + x.false)).fillna(
        0
    )

    mean_detecs = first_detecs.mean()
    df_metrics.loc["mean"] = mean_detecs
    mean_true_detecs = first_detecs.apply(
        lambda x: [a - drift_start if a >= drift_start else None for a in x]
    ).mean()
    df_metrics.loc["mean_true"] = mean_true_detecs

    median_detecs = first_detecs.median()
    df_metrics.loc["median"] = median_detecs
    median_true_detecs = first_detecs.apply(
        lambda x: [a - drift_start if a >= drift_start else None for a in x]
    ).median()
    df_metrics.loc["median_true"] = median_true_detecs

    max_detecs = first_detecs.max()
    df_metrics.loc["max"] = max_detecs
    max_true_detecs = first_detecs.apply(
        lambda x: [a - drift_start if a >= drift_start else None for a in x]
    ).max()
    df_metrics.loc["max_true"] = max_true_detecs

    min_detecs = first_detecs.min()
    df_metrics.loc["min"] = min_detecs
    min_true_detecs = first_detecs.apply(
        lambda x: [a - drift_start if a >= drift_start else None for a in x]
    ).min()
    df_metrics.loc["min_true"] = min_true_detecs

    std_detecs = first_detecs.std()
    df_metrics.loc["std"] = std_detecs
    std_true_detecs = first_detecs.apply(
        lambda x: [a - drift_start if a >= drift_start else None for a in x]
    ).std()
    df_metrics.loc["std_true"] = std_true_detecs
    return df_metrics


def get_detecs(df_results):
    dict_detects = {}
    for col in df_results.columns:
        dict_detects[col] = df_results.apply(
            lambda x: [a[0] if len(a) > 0 else np.nan for a in x], axis=0
        )[col].unique()
        dict_detects[col] = [x for x in dict_detects[col] if not np.isnan(x)]
    return dict_detects


def get_true_false_detecs(dict_detects, D_G):
    true_detecs_dict, false_detecs_dict = {}, {}
    for k, v in dict_detects.items():
        true_detecs_dict[k] = [x for x in v if x >= D_G.drifts[0].start]
        false_detecs_dict[k] = [x for x in v if x < D_G.drifts[0].start]
    return (true_detecs_dict, false_detecs_dict)


def get_df_detections(
    D_G,
    results,
    selected_methods,
    exp_type="model_reset",
    n_points=None,
    noisy=False,
    path=None,
):
    all_df_results = []
    current_index = 0
    sep_bar_indexes = []
    if n_points is None:
        n_points = D_G.n
    results = list(filter(lambda x: str(n_points) in x, results))
    dataset_results = list(
        filter(lambda x: D_G.drift_name.replace(" ", "_") in x, results)
    )
    dataset_results = list(filter(lambda x: str(D_G.n) in x, dataset_results))

    if noisy and D_G.noise_rate != 0:
        dataset_results = list(filter(lambda x: "noisy" in x, dataset_results))
        dataset_results = list(
            filter(lambda x: str(D_G.noise_rate)[2:] in x, dataset_results)
        )
    elif noisy and D_G.noise_rate == 0:
        dataset_results = list(filter(lambda x: "_noisy" not in x, dataset_results))
        dataset_results = list(
            filter(lambda x: str(D_G.noise_rate)[2:] in x, dataset_results)
        )
    else:
        dataset_results = list(filter(lambda x: "noisy" not in x, dataset_results))
    for method in selected_methods:
        for res in [x for x in dataset_results if method in x]:
            if str(D_G.n) in res and exp_type in res:
                if "noisy" in res:
                    df_results = get_df_results(res, path=path)
                    if D_G.noise_rate == 0:
                        df_results.columns = [f"{x}" for x in df_results.columns]
                    else:
                        df_results.columns = [
                            f"{x}_{D_G.noise_rate}" for x in df_results.columns
                        ]
                else:
                    df_results = get_df_results(res, path=path)

                all_df_results.append(df_results.reset_index(drop=True))
                current_index += 1 * len(df_results.columns)
        sep_bar_indexes.append(current_index)
    print([x.shape for x in all_df_results])
    df_results_dataset = pd.concat(all_df_results, axis=1)
    return (df_results_dataset, sep_bar_indexes[:-1])


def get_D_G_metrics(
    D_G,
    results,
    selected_methods=["ADWIN", "PH", "KSWIN"],
    exp_type="model_reset",
    noisy=True,
    path=None,
    verbose=False,
):
    # if (D_G.noise_rate == 0):
    #    noisy = False
    if verbose:
        print(f"Giving results for {D_G.drift_name} with {D_G.n} points - {exp_type}")
    all_df_metrics = []
    dataset_results = list(
        filter(lambda x: D_G.drift_name.replace(" ", "_") in x, results)
    )
    dataset_results = list(filter(lambda x: str(D_G.n) in x, dataset_results))
    if noisy and D_G.noise_rate != 0:
        dataset_results = list(filter(lambda x: "noisy" in x, dataset_results))
        dataset_results = list(
            filter(lambda x: str(D_G.noise_rate)[2:] in x, dataset_results)
        )
    elif noisy and D_G.noise_rate == 0:
        temp_save = dataset_results.copy()
        dataset_results = list(filter(lambda x: "_noisy_" in x, dataset_results))

        dataset_results = list(
            filter(lambda x: str(D_G.noise_rate)[2:] in x, dataset_results)
        )
        if len(dataset_results) == 0:
            dataset_results = list(filter(lambda x: "noisy" not in x, temp_save))
            dataset_results = list(
                filter(lambda x: str(D_G.noise_rate)[2:] in x, dataset_results)
            )
    else:
        dataset_results = list(filter(lambda x: "noisy" not in x, dataset_results))

    for method in selected_methods:
        #    df_results = pd.read_csv(results_path+"abrupt_concept_drift_5000_df_reset_ADWIN.csv", index_col=0)
        for res in [x for x in dataset_results if method in x]:
            if D_G.noise_rate == 0 and exp_type == "noisy":
                exp_type = ""
            if str(D_G.n) in res and exp_type in res:
                if "noisy" in res:
                    df_results = get_df_results(res, path=path)
                    if D_G.noise_rate == 0:
                        df_results.columns = [f"{x}" for x in df_results.columns]
                    else:
                        df_results.columns = [
                            f"{x}_{D_G.noise_rate}" for x in df_results.columns
                        ]
                else:
                    df_results = get_df_results(res, path=path)
                if len(D_G.drifts) > 1 and D_G.drifts[0].start == 0:
                    all_df_metrics.append(
                        get_df_metrics(df_results, D_G.drifts[1].start)
                    )
                else:
                    all_df_metrics.append(
                        get_df_metrics(df_results, D_G.drifts[0].start)
                    )
    if len(all_df_metrics) == 0:
        print("     No results for this request")
        print(path, D_G.drift_name, D_G.noise_rate)
    else:
        all_df_metrics = pd.concat((all_df_metrics), axis=1)
    return all_df_metrics


def get_expe_list(dataset_name, results, verbose=False):
    df_results = [x for x in results if dataset_name in x]
    df_results = [x for x in df_results if len([i for i in x if i.isdigit()]) > 0]
    try:
        n_points = np.unique(
            [[int(s) for s in x.split("_") if s.isdigit()] for x in df_results]
        )
        if len([x for x in n_points if len(str(x)) > 3]) > 0:
            # print()
            if verbose:
                print(f"{dataset_name} _ {n_points}")
            return list(n_points)
    except:
        pass
    return df_results


def get_dataset_list(results):
    remove_list = ["ADWIN", "PH", "KSWIN", "_df_reset", "_model_reset"]
    aaa = [x.split(".csv")[0] for x in results]
    aaa = [x for x in aaa if len([i for i in x if i.isdigit()]) > 0]
    for word in remove_list:
        aaa = list(map(lambda x: x.replace(word, ""), aaa))
    aaa = [s.replace("__", "") for s in aaa]
    aaa = np.unique([s[:-1] if s[-1] == "_" else s for s in aaa])
    for i in range(3):
        # not noisy case
        aaa = [
            "_".join(x.split("_")[:-1]) if x[-1].isdigit() and not "noisy" in x else x
            for x in aaa
        ]
        # noisy case
        aaa = [
            "_".join(x.split("_")[:-1])
            if x.split("_")[-1].isdigit() and "noisy" in x
            else x
            for x in aaa
        ]
        aaa = [x for x in aaa if len(x) > 0]
    return list(np.unique(aaa))


def res_to_latex(res):
    df = res.copy()

    metrics_list = ["no", "false", "min", "mean", "median", "median_true", "max", "std"]
    df = df.loc[metrics_list].T.sort_values(["no", "false", "median"])  # .to_latex()

    #    df.loc[:,"mean"] = df.loc[:,["mean", "median", "median_true"]].apply(lambda x: x[0]-(x[1]-x[2]), axis=1)
    df.loc[:, ["no"]] = 1 - df.loc[:, ["no"]]
    # df = df.rename(columns = {"no":"d_rate"})
    df.loc[:, ["no", "false"]] = df.loc[:, ["no", "false"]].applymap("{0:.2f}".format)
    df.loc[:, ["mean", "median", "median_true", "std"]] = df.loc[
        :, ["mean", "median", "median_true", "std"]
    ].applymap("{0:.0f}".format)

    df = df.drop(columns=["min", "max", "median_true"])
    df = df.rename(columns={"median": "med", "no": "d_rate", "false": "FP"})
    # df.loc[:,["no", "false"]] = df.loc[:,["no", "false"]].applymap("{0:.0f}".format)
    text = df.to_latex()
    text = text.replace("retrain\_", "")

    text = text.replace("toprule", "hline")
    text = text.replace("llllll", "|l|l|l|l|l|l|")
    text = text.replace("midrule", "")
    text = text.replace("bottomrule", "hline")
    text = text.replace(" \\", " \\\\\hline")
    text = text.replace("hline\\", "hline")
    text = text.replace("hline\n\end", "\n\end")
    text = text.replace("\\\n", "\n")
    text = text.replace("_loss", "")

    print(text)
