import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from .bench_fcts import get_df_metrics, get_detecs, get_true_false_detecs, get_df_detections


# TODO: update this func
def pre_compute_plot(self, error_list, detections, funcs, w_size=500):
    # pre-compute error means
    error_means_list = [[] for i in range(len(funcs))]

    for k in range(np.shape(error_list)[0]):
        for i, (err, label) in enumerate(zip(error_list[k], funcs)):
            if (len(err) > 0):
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
            if (len(detections_interval) > 0):
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

    return (error_means_list, drift_points, detecs, stats_detections, freq_detections)


def plot_circles(self):
    if (self.n_features == 2):
        cmap = get_cmap('tab10')

        fix, ax = plt.subplots(nrows=1, ncols=1, figsize=[20, 10])
        for i, (c_x, c_y) in enumerate(self.circle_centers):  # drift_centers):
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
        [(func, stats[i][1][0]) for i, func in enumerate(fcts)], key=lambda x: x[1], reverse=True)]
    i_ordered = [list(fcts).index(x) for x in o_funcs if x in fcts]
    stats = np.array(stats)[i_ordered]
    freq_detec_ordered = np.array(freq_detections)[i_selected][i_ordered]

    if (ax == None):
        fig, ax = plt.subplots(nrows=len(drift_points)-1,
                               ncols=1, figsize=[20, 2*len(fcts)])

    colors = ['b', 'g', 'purple', 'cyan',
              'lime', 'palegoldenrod', 'navy', 'flame']
    if (len(funcs) > 8):
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
            if (mean > 0):

                ax[d].vlines(ymin=ymin, ymax=ymax, x=mean+self.n_seen(),
                             color=colors[i], label="%.2f : %s | + %d" % (freq_detec_ordered[i][d], func[8:], int(mean-d_p)))
                ax[d].fill_betweenx(y=[ymin, ymax], x1=max(early, mean - var)+self.n_seen(),
                                    x2=min(drift_points[d+1], mean + var)+self.n_seen(), alpha=0.35, color=colors[i])
            else:
                ax[d].plot([], [], ' ', label="0 : "+func[8:])
            if (early > 0):
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
        [(func, stats[i][1][0]) for i, func in enumerate(fcts)], key=lambda x: x[1], reverse=True)]
    i_ordered = [list(fcts).index(x) for x in o_funcs if x in fcts]
    stats = np.array(stats)[i_ordered]
    freq_detec_ordered = np.array(freq_detections)[i_selected][i_ordered]
    o_detecs = np.array(detecs, dtype='object')[i_selected][i_ordered]

    ax_sizes = [sum([len(x) > 0 for x in o_detecs[:, i]])
                for i in range(np.shape(o_detecs)[1])]
    ax_sizes = [x+1 if x == 0 else x for x in ax_sizes]

    if (ax == None):
        fig, ax = plt.subplots(nrows=np.shape(o_detecs)[1], ncols=1,
                               figsize=[20, sum(ax_sizes)*0.75], gridspec_kw={'height_ratios': ax_sizes})

    colors = ['orange', 'cyan', 'crimson', 'lime', 'violet', 'dodgerblue',
              'lightslategrey', 'olivedrab']
    if (len(funcs) > 8):
        colors = [[x for x in colors][i %
                                      len(colors)] for i in range(len(funcs))]

    ymin_max = [(i)/len(fcts) for i in range(len(fcts)+1)]

    for d, d_p in enumerate(drift_points[:-1]):
        ax[d].vlines(ymin=0, ymax=1, x=d_p, color='red', label="drift point")
        ax[d].vlines(ymin=0, ymax=1, x=drift_points[d+1],
                     color='red', label="drift point")

    for d, d_p in enumerate(drift_points[:-1]):

        index_detecs = [i for i, x in enumerate(
            freq_detec_ordered[:, d]) if x > 0]

        if (len(index_detecs) > 0):
            func_detecs = o_detecs[index_detecs, d]

            violins = ax[d].violinplot(func_detecs, vert=False, showmeans=True,
                                       widths=1.2/(ax_sizes[d]+1),
                                       positions=[(i)/(ax_sizes[d]+1) for i in range(1, ax_sizes[d]+1)])
            # for ((i,func), v) in zip(enumerate(o_funcs),violins['bodies']):
            for i, (ind, func, v) in enumerate(zip(index_detecs, np.array(o_funcs)[index_detecs], violins['bodies'])):
                v.set_fc(c=colors[ind])
                if (freq_detec_ordered[i][d] < 0.98):
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

    if (ax == None):
        fig, ax = plt.subplots(nrows=len(funcs), ncols=1,
                               figsize=[20, 3*len(funcs)])

    for i, func in enumerate(funcs):
        mean = np.mean(error_means_list[i], axis=0)
        var = np.std(error_means_list[i], axis=0)
        ax[i].plot(mean, label="mean-error")
        ax[i].fill_between(np.arange(len(mean)), mean -
                           var, mean+var, alpha=0.30)
        ax[i].set_title("%.3f  " % (np.mean(mean))+func[8:], fontsize=20)

        if (ymin == None):
            ymin = 0.85
        if (ymax == None):
            ymax = 1

        for d in self.drift_points:
            ax[i].vlines(ymin=ymin, ymax=ymax, x=d - self.n_seen(),
                         color='red', label="drift point")

        for d_p in range(len(drift_points)-1):
            mean, var, count, early = stats_detections[i][d_p]
            if (mean != 0):
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

    if (ax == None):
        fig, ax = plt.subplots(nrows=len(funcs), ncols=1,
                               figsize=[20, 4*len(funcs)])
    detections = np.array(detections, dtype=object)
    n_seen = self.n_train+self.n_test

    for k in range(np.shape(error_means_list)[1]):
        for i, label in enumerate(funcs):
            error_mean = error_means_list[i][k]
            ax[i].plot(np.arange(n_seen, n_seen+len(error_mean)), error_mean)
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

    if (ax == None):
        fig, ax = plt.subplots(nrows=len(fcts), ncols=1,
                               figsize=[20, 4*len(fcts)])
    detections = np.array(detections, dtype=object)
    n_seen = self.n_train+self.n_test

    for k in range(np.shape(error_means_list)[1]):
        for i, label in enumerate(fcts):
            error_mean = error_means_list[i][k]
            ax[i].plot(np.arange(n_seen, n_seen+len(error_mean)), error_mean)
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

    if (ax == None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[20, 12])

    for i, func in enumerate(funcs):
        if (func == "retrain_shap_adwin"):
            func = "retrain_shap_adwin_random"
        mean = np.mean(error_means_list[i], axis=0)
        var = np.std(error_means_list[i], axis=0)
        ax.plot(mean, label="%.3f  " % np.mean(mean)+func[8:])
        ax.fill_between(np.arange(len(mean)), mean-var, mean+var, alpha=0.30)

        ax.set_ylim(ymin, ymax)

    # sort legend
    labels = ax.get_legend_handles_labels()
    labels = [(a, b, float(b[:5])) for a, b in zip(labels[0], labels[1])]
    labels = sorted(labels, key=lambda x: x[2], reverse=True)
    handles, labels = [x[0] for x in labels], [x[1] for x in labels]

    plt.legend(handles, labels, fontsize=20)


def plot_evolving_noise_rates(df, log_scale=True):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=[
                           10, 5])  # df.algo.unique().size/2

    df_mean = df.groupby(["algo", "noise_rate"]).mean()
    # print(df_mean)
    for i, algo in enumerate(df.algo.unique()):
        no = 1 - df_mean[["no"]].loc[algo].values
        false = df_mean[["false"]].loc[algo].values
        # print(no, false)
        # print(df.noise_rate.unique())
        noise_rate = [float(x) for x in df.noise_rate.unique()]
        ax[i % 2, i//2].fill_between(noise_rate, np.ravel(no-false), np.ravel(
            no), label="False detection", color='red')  # sns.color_palette("Paired")[2*i+1]
        ax[i % 2, i//2].fill_between(noise_rate, np.ravel(no-false),
                                     np.zeros(len(no)), label="True detection", color='green')
        if (log_scale):
            # print(ax[i % 2, i//2].get_xticks())
            ax[i % 2, i//2].set_xscale('log')
            ticks = ax[i % 2, i//2].get_xticks()[:]
            ticks[-1] = max(noise_rate)
            ticks = [x for x in ticks if 0 < x <= max(noise_rate)]
            ticks_labels = ticks.copy()
            ticks_labels[-1] = max(noise_rate)
            if (0 in noise_rate):
                ticks_labels[1] = 0
            ax[i % 2, i//2].set_xticks(ticks=ticks, labels=ticks_labels)
            ax[i % 2, i//2].set_xlim(sorted(noise_rate)[1]/10, max(noise_rate))

        ax[i % 2, i//2].set_ylim(0, 1)
        ax[i % 2, i//2].set_title(" ".join([x.capitalize()
                                  for x in algo.replace("_loss", "").split("_")]), fontsize=20)

        if (i % 2 != 1):
            ax[i % 2, i//2].set_xticks([])

    fig.text(0.08, 0.5, 'Detection Rate', ha='center',
             va='center', rotation='vertical',  fontsize=20)
    fig.text(0.5, 0.02, 'Label Noise Rate', ha='center',
             va='center', rotation='horizontal',  fontsize=20)

    plt.subplots_adjust(wspace=0.2)
    return (ax)


def plot_evolving_perf(df):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=[
                           10, 5])  # df.algo.unique().size/2
    cmap1 = LinearSegmentedColormap.from_list('Custom', [x for i, x in enumerate(
        sns.color_palette("tab20")) if i % 2 == 0][:4], 4)
    cmap2 = LinearSegmentedColormap.from_list('Custom', [x for i, x in enumerate(
        sns.color_palette("tab20")) if i % 2 == 1][:4], 4)
    for k in range(2):
        for i, algo in enumerate(df.algo.unique()):
            # if(i%2==0):
            #    df[df.algo == algo].groupby('noise_rate').mean(numeric_only=True).loc[:,
            #            ["median", "std"]].plot(ax=ax[i//2], cmap=cmap1)#["median","max","median_true", "std"]
            # else:
            #    df[df.algo == algo].groupby('noise_rate').mean(numeric_only=True).loc[:,
            #            ["get_df_metrics", "std"]].plot(ax=ax[i//2], cmap=cmap2)
            #
            vals_df = df[df.algo == algo].groupby('noise_rate').mean(numeric_only=True).loc[:,
                                                                                            ["median", "std", "median_true"]]
            noise_rate = vals_df.index.values
            med, std = vals_df.loc[:,
                                   "median"].values, vals_df.loc[:, "std"].values
            if (k == 0):
                val = vals_df.loc[:, "median"].values
            else:
                val = vals_df.loc[:, "std"].values

            # if("shap" in algo):
            #    ax[k, i//2].plot(noise_rate, val, label=algo+"_shap")
            # else:
            #    ax[k, i//2].plot(noise_rate, val, label="median_detect")
            ax[k, i//2].plot(noise_rate, val,
                             label=algo.replace("_loss", "").capitalize())

            # ax[k, i//2].fill_between(noise_rate, med-std, med+std, alpha=0.3)
            if (k == 0):
                ax[k, i//2].set_ylim(-10, 4000)
            else:
                ax[k, i//2].set_ylim(0, 2000)
            if (k == 0):
                ax[k, i//2].set_title(" ".join([x.capitalize() for x in
                                                algo.replace("_loss", "").replace("_shap", "").split("_")]), fontsize=20)
            else:
                ax[k, i//2].set_xticks([])
            ax[k, i//2].set_xlabel("")
            ax[k, i//2].legend(fontsize=8)

    ax[0, 0].set_ylabel("Median detection")
    ax[1, 0].set_ylabel("std")
#    fig.text(0.06, 0.5, 'Detection Rate', ha='center', va='center', rotation='vertical',  fontsize =20)
    fig.text(0.5, 0.03, 'Label Noise Rate', ha='center',
             va='center', rotation='horizontal',  fontsize=20)

    plt.subplots_adjust(wspace=0.3)
    return (fig, ax)


def plot_median_evolution(df, selected_methods):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15, 5])
    all_algos = np.unique(df.algo)
    for m in selected_methods:
        selected_algos = all_algos[[i for (i, x) in enumerate(
            all_algos) if m.lower() in x.lower()]]

        for algo in selected_algos:
            med = df.groupby(["algo", "noise_rate"]).mean()[
                ["median"]].loc[algo].values
            std = df.groupby(["algo", "noise_rate"]).mean()[
                ["std"]].loc[algo].values
            std = np.array([[x[0]] if not np.isnan(x[0]) else [0]
                           for x in std])
            noise_rate = [float(x) for x in df.noise_rate.unique()]
            ax.plot(noise_rate, med, label=algo+"+std")
            lower_bound = np.ravel(med-std)
            upper_bound = np.ravel(med+std)
            ax.fill_between(noise_rate, lower_bound, upper_bound, alpha=0.3)

        plt.legend()
        ax.set_ylabel("Median detection ")
    return (ax)


def plot_violins_shap_noise(single_drift, D_G, selected_methods, results, path=None, set_same_xrange=True):

    for i, f in enumerate(single_drift[:]):
        D_G = f()
        # print("#"*20, f"{D_G.drift_name} {D_G.n}", "#"*20)

        if (D_G.noise_rate != 0):
            df_results_dataset, sep_bar_indexes = get_df_detections(D_G, results,
                                                                    selected_methods, exp_type="df_reset", path=path, noisy=True)
        else:
            df_results_dataset, sep_bar_indexes = get_df_detections(D_G, results,
                                                                    selected_methods, exp_type="df_reset", path=path, noisy=False)
        for col in ["retrain_PH_loss", "retrain_KSWIN_0", "retrain_KSWIN.", "retrain_adwin_loss"]:
            for real_col in df_results_dataset.columns:
                if (col in real_col):
                    #    if(col in df_results_dataset.columns):
                    df_results_dataset = df_results_dataset.drop(columns=[
                                                                 real_col])

        if (len(df_results_dataset.columns) == 1):
            df_results_dataset.loc[:, df_results_dataset.columns[0].replace(
                "_shap_", "_")] = [[] for x in range(len(df_results_dataset))]
        if (i == 0):
            df_res_sing_drift = df_results_dataset
        else:
            df_res_sing_drift = pd.concat(
                [df_res_sing_drift, df_results_dataset], axis=1)

    # sort cols by mean detect

    df_res_sing_drift.columns = list(
        map(lambda x: x+"_0.0" if "_0" not in x else x, df_res_sing_drift.columns))
    dict_detects_temp = get_detecs(df_res_sing_drift)
    dict_detects = {}
    for k, v in dict_detects_temp.items():
        if (len(v) > 0):
            dict_detects[k] = int(np.mean(v))
        else:
            dict_detects[k] = int(D_G.n)
    # for k in ['retrain_KSWIN_shap_0.1', 'retrain_KSWIN_shap_0.05', 'retrain_KSWIN_shap_0.0', 'retrain_KSWIN_shap_0.01', 'retrain_KSWIN_shap_0.001', 'retrain_KSWIN_0.1', 'retrain_KSWIN_0.05', 'retrain_KSWIN_0.0', 'retrain_KSWIN_0.01', 'retrain_KSWIN_0.001', 'retrain_KSWIN_0.0', 'retrain_KSWIN_shap_0.5']:
    #    if k not in dict_detects:
    #        dict_detects[k]=int(D_G.n)
    sorted_k = list({k: v for k, v in sorted(
        dict_detects.items(), key=lambda item: item[1])})

    # print(df_res_sing_drift)

    df_res_sing_drift = df_res_sing_drift[sorted_k]
    df_res_sing_drift_s = df_res_sing_drift[[
        x for x in df_res_sing_drift.columns if "shap" not in x]]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[15, 5])
    # plot_violins_shap_noise(single_drift, D_G, selected_methods, results, ax=ax[0])
    colors = [sns.color_palette("Paired")[(2*i+1) % 12]
              for i in range(len(single_drift))]
    _ = plot_violins(D_G, df_res_sing_drift_s, ax=ax[0], sep_bar_indexes=sep_bar_indexes,
                     separate_true_false=False,
                     colors=colors)

    df_res_sing_drift_s = df_res_sing_drift[[
        x for x in df_res_sing_drift.columns if "shap" in x]]

    a = [x.split("_")[-1] for x in sorted_k if "shap" not in x]
    # a = list(map(lambda x: "0.0" if x in ["shap","adwin","KSWIN", "PH"] else x, a))
    b = [x.split("_")[-1] for x in sorted_k if "shap" in x]
    # b = list(map(lambda x: "0.0" if x in ["shap","adwin","KSWIN", "PH"] else x, b)
    # colors = [sns.color_palette("Paired")[(2*i+1) % 12] for i in range(len(single_drift))]
    # colors = [sns.color_palette("Paired")[(2*i+1) % 12] for i in [a.index(x) for x in b]]

    _ = plot_violins(D_G, df_res_sing_drift_s, ax=ax[1], sep_bar_indexes=sep_bar_indexes,
                     separate_true_false=False, path=path,
                     colors=colors)
    ax[0].set_title(selected_methods[0], fontsize=30)
    ax[1].set_title(selected_methods[0]+"_SHAP", fontsize=30)

    if (set_same_xrange):
        xmin_0, xmax_0 = ax[0].get_xlim()
        xmin_1, xmax_1 = ax[1].get_xlim()
        xmin = min(xmin_0, xmin_1)
        xmax = max(xmax_0, xmax_1)
        ax[0].set_xlim(xmin, xmax)
        ax[1].set_xlim(xmin, xmax)

    handles, previous_labels = ax[0].get_legend_handles_labels()
    new_labels = [x.split("|")[0] for x in previous_labels]
    ax[0].legend(handles=handles[::-1], labels=new_labels[::-1])
    ax[0].set_yticklabels([x.split("|")[-1].split("_")[-1]
                          for x in previous_labels][1:])
    ax[0].set_yticks(np.linspace(0+1/(2*len(single_drift)), 1-1/(2*len(single_drift)),
                     len(single_drift)), [x.split("|")[-1].split("_")[-1] for x in previous_labels][1:])
    ax[0].set_ylabel("Noise Rate", fontsize=22)
    ax[0].set_xlabel("Point index", fontsize=22)

    handles, previous_labels = ax[1].get_legend_handles_labels()
    print(previous_labels)
    new_labels = [x.split("|")[0] for x in previous_labels]
    ax[1].legend(handles=handles[::-1], labels=new_labels[::-1])
    ax[1].set_yticks(np.linspace(0+1/(2*len(single_drift)), 1-1/(2*len(single_drift)),
                     len(single_drift)), [x.split("|")[-1].split("_")[-1] for x in previous_labels][1:])
    # ax[1].set_ylabel("Noise Rate", fontsize = 22)
    ax[1].set_xlabel("Point index", fontsize=22)

    # ax[0].legend([])
    # ax[1].legend([])

    ax[0].tick_params(axis='both', which='major', labelsize=18)
    ax[1].tick_params(axis='both', which='major', labelsize=18)

    plt.subplots_adjust(wspace=0.2)
    return (fig, ax)




def plot_violins_shap_noise_TIGHT(single_drift, D_G, selected_methods, results, path=None, set_same_xrange=True):

    for i, f in enumerate(single_drift[:]):
        D_G = f()
        # print("#"*20, f"{D_G.drift_name} {D_G.n}", "#"*20)

        if (D_G.noise_rate != 0):
            df_results_dataset, sep_bar_indexes = get_df_detections(D_G, results,
                                                                    selected_methods, exp_type="df_reset", path=path, noisy=True)
        else:
            df_results_dataset, sep_bar_indexes = get_df_detections(D_G, results,
                                                                    selected_methods, exp_type="df_reset", path=path, noisy=False)
        for col in ["retrain_PH_loss", "retrain_KSWIN_0", "retrain_KSWIN.", "retrain_adwin_loss"]:
            for real_col in df_results_dataset.columns:
                if (col in real_col):
                    #    if(col in df_results_dataset.columns):
                    df_results_dataset = df_results_dataset.drop(columns=[
                                                                 real_col])

        if (len(df_results_dataset.columns) == 1):
            df_results_dataset.loc[:, df_results_dataset.columns[0].replace(
                "_shap_", "_")] = [[] for x in range(len(df_results_dataset))]
        if (i == 0):
            df_res_sing_drift = df_results_dataset
        else:
            df_res_sing_drift = pd.concat(
                [df_res_sing_drift, df_results_dataset], axis=1)

    # sort cols by mean detect

    df_res_sing_drift.columns = list(
        map(lambda x: x+"_0.0" if "_0" not in x else x, df_res_sing_drift.columns))
    dict_detects_temp = get_detecs(df_res_sing_drift)
    dict_detects = {}
    for k, v in dict_detects_temp.items():
        if (len(v) > 0):
            dict_detects[k] = int(np.mean(v))
        else:
            dict_detects[k] = int(D_G.n)
    # for k in ['retrain_KSWIN_shap_0.1', 'retrain_KSWIN_shap_0.05', 'retrain_KSWIN_shap_0.0', 'retrain_KSWIN_shap_0.01', 'retrain_KSWIN_shap_0.001', 'retrain_KSWIN_0.1', 'retrain_KSWIN_0.05', 'retrain_KSWIN_0.0', 'retrain_KSWIN_0.01', 'retrain_KSWIN_0.001', 'retrain_KSWIN_0.0', 'retrain_KSWIN_shap_0.5']:
    #    if k not in dict_detects:
    #        dict_detects[k]=int(D_G.n)
    sorted_k = list({k: v for k, v in sorted(
        dict_detects.items(), key=lambda item: item[1])})

    # print(df_res_sing_drift)

    df_res_sing_drift = df_res_sing_drift[sorted_k]
    df_res_sing_drift_s = df_res_sing_drift[[
        x for x in df_res_sing_drift.columns if "shap" not in x]]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[10, 4])
    # plot_violins_shap_noise(single_drift, D_G, selected_methods, results, ax=ax[0])
    colors = [sns.color_palette("Paired")[(2*i+1) % 12]
              for i in range(len(single_drift))]
    _ = plot_violins(D_G, df_res_sing_drift_s, ax=ax[0], sep_bar_indexes=sep_bar_indexes,
                     separate_true_false=False,
                     colors=colors, plot_single_detects=False)

    df_res_sing_drift_s = df_res_sing_drift[[
        x for x in df_res_sing_drift.columns if "shap" in x]]

    a = [x.split("_")[-1] for x in sorted_k if "shap" not in x]
    # a = list(map(lambda x: "0.0" if x in ["shap","adwin","KSWIN", "PH"] else x, a))
    b = [x.split("_")[-1] for x in sorted_k if "shap" in x]
    # b = list(map(lambda x: "0.0" if x in ["shap","adwin","KSWIN", "PH"] else x, b)
    # colors = [sns.color_palette("Paired")[(2*i+1) % 12] for i in range(len(single_drift))]
    # colors = [sns.color_palette("Paired")[(2*i+1) % 12] for i in [a.index(x) for x in b]]

    _ = plot_violins(D_G, df_res_sing_drift_s, ax=ax[1], sep_bar_indexes=sep_bar_indexes,
                     separate_true_false=False, path=path,
                     colors=colors, plot_single_detects=False)
    ax[0].set_title(selected_methods[0], fontsize=30)
    ax[1].set_title(selected_methods[0]+"_SHAP", fontsize=30)

    if (set_same_xrange):
        xmin_0, xmax_0 = ax[0].get_xlim()
        xmin_1, xmax_1 = ax[1].get_xlim()
        xmin = min(xmin_0, xmin_1)
        xmax = max(xmax_0, xmax_1)
        ax[0].set_xlim(xmin, xmax)
        ax[1].set_xlim(xmin, xmax)

    handles, previous_labels = ax[0].get_legend_handles_labels()
    new_labels = [x.split("|")[0] for x in previous_labels]
    ax[0].legend(handles=handles[::-1], labels=new_labels[::-1], loc="best")
    ax[0].set_yticklabels([x.split("|")[-1].split("_")[-1]
                          for x in previous_labels][1:])
    ax[0].set_yticks(np.linspace(0+1/(2*len(single_drift)), 1-1/(2*len(single_drift)),
                     len(single_drift)), [x.split("|")[-1].split("_")[-1] for x in previous_labels][1:])
    ax[0].set_ylabel("Noise Rate", fontsize=22)
    
    

    handles, previous_labels = ax[1].get_legend_handles_labels()
    #print(previous_labels)
    new_labels = [x.split("|")[0] for x in previous_labels]
    ax[1].legend(handles=handles[::-1], labels=new_labels[::-1], loc="best")
    ax[1].set_yticks(np.linspace(0+1/(2*len(single_drift)), 1-1/(2*len(single_drift)),
                     len(single_drift)), [x.split("|")[-1].split("_")[-1] for x in previous_labels][1:])
    ax[1].set_yticks([],[])
    # ax[1].set_ylabel("Noise Rate", fontsize = 22)
    ax[1].set_xlabel("Point index", fontsize=22)

    #ax[0].legend([])
    #ax[1].legend([])

    ax[0].tick_params(axis='both', which='major', labelsize=18)
    ax[1].tick_params(axis='both', which='major', labelsize=18)
    

    
    plt.subplots_adjust(wspace=0)

    xlabels = [item.get_text() for item in ax[0].get_xticklabels()]
    print(xlabels)
    xlabels[-1] = ''
    xlabels[-2] = ''
    ax[0].set_xticklabels(xlabels)
    ax[0].set_xlabel("Point index", fontsize=22)
    return (fig, ax)


def plot_violins(D_G, df_results, ax=None, sep_bar_indexes=None,
                 separate_true_false=True, save=False, path=None,
                 colors=None, plot_single_detects=True):  # TODO: Fix when there is an uneven number of results columns (separation bars are not correct)
    df_metrics = get_df_metrics(df_results, drift_start=D_G.drifts[0].start)

    dict_detects = get_detecs(df_results)

    true_detecs_dict, false_detecs_dict = get_true_false_detecs(
        dict_detects, D_G)

    if (ax == None):
        fig, ax = plt.subplots(nrows=1, ncols=1,
                               figsize=[15, len(df_results.columns)*1.15])

    for d, d_p in enumerate([D_G.drifts[0].start]):
        ax.vlines(ymin=0, ymax=1, x=d_p, color='red',
                  label="drift start", linestyles="dashed")

    for i, col in enumerate(sorted(df_results.columns)):
        # f_d, no_d, precision = df_metrics.loc[["false", "no"], col].values
        _, _, precision, TP = df_metrics.loc[[
            "false", "no", "prec", "TP"], col].values
        if (colors == None):
            color = sns.color_palette("Paired")[(2*i+1) % 12]
        else:
            color = colors[i]
        label = f"TP:{TP:.2f}|"+col.split("retrain_")[1]
        # label = f"detec {1-no_d:.2f}|FD {f_d:.2f}|"+col.split("retrain_")[1]
        ax.plot([], [], color=color, label=label)
        if (separate_true_false):
            if (len(true_detecs_dict[col]) > 0):
                violins = ax.violinplot(true_detecs_dict[col], vert=False, showmedians=True,
                                        widths=1/len(df_results.columns),
                                        positions=[
                                            i/len(df_results.columns)+(1/len(df_results.columns))/2],
                                        points=1000,)
                for v in violins["bodies"]:
                    v.set_fc(color)
                    v.set_alpha(0.8)
                for arg in ['cmaxes', 'cmins', 'cmedians', 'cbars']:
                    violins[arg].set_edgecolor('black')
        # for i, col in enumerate(df_results.columns):#TODO match colors with Paired or something like that
            if (len(false_detecs_dict[col]) > 0):
                violins = ax.violinplot(false_detecs_dict[col], vert=False, showmedians=True,
                                        widths=1/len(df_results.columns),
                                        positions=[
                                            i/len(df_results.columns)+(1/len(df_results.columns))/2],
                                        points=1000)
                v = violins["bodies"][0]
                color = sns.color_palette("Paired")[(2*i) % 12]
                v.set_fc(color)
                v.set_alpha(0.8)
                for arg in ['cmaxes', 'cmins', 'cmedians', 'cbars']:
                    violins[arg].set_edgecolor('black')
        else:
            if (len(dict_detects[col]) > 1):
                violins = ax.violinplot(dict_detects[col], vert=False, showmedians=True,
                                        widths=1/len(df_results.columns),
                                        positions=[
                                            i/len(df_results.columns)+(1/len(df_results.columns))/2],
                                        points=1000,)
                for v in violins["bodies"]:
                    v.set_fc(color)
                    v.set_alpha(0.8)
                for arg in ['cmaxes', 'cmins', 'cmedians', 'cbars']:
                    violins[arg].set_edgecolor('black')

        detecs = [x for x in dict_detects[col] if x > 0]
        if plot_single_detects:
            for d in detecs:
                ymax = i/len(df_results.columns)+(1/len(df_results.columns)
                                                  )/2+(1/len(df_results.columns)/2 + 0.01)
                ymin = i/len(df_results.columns)+(1/len(df_results.columns)
                                                  )/2-(1/len(df_results.columns)/2 + 0.01)
                # print(ymin, ymax)
                ax.vlines(ymin=ymin, ymax=ymax, x=d,
                          color='orange', alpha=1, zorder=-1)

    if (sep_bar_indexes is not None):
        xmin, xmax = ax.get_xlim()
        for i in sep_bar_indexes:
            y = ((i-1))/len(df_results.columns) + \
                (1/len(df_results.columns))/2+(1/len(df_results.columns)/2)
            ax.hlines(xmin=xmin, xmax=xmax, y=y,
                      color='black', alpha=0.3, zorder=-1)

    # invert legend order
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(reversed(labels), reversed(handles)))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    return (ax)


def plot_detector_param_noise(df_perf_detector, param_name_str, detector_name, default_perf_if_fail):
    if ("delta" in param_name_str):
        beautiful_param_name = r"$\delta$"
    elif ("alpha" in param_name_str):
        beautiful_param_name = r"$\alpha$"
    else:
        beautiful_param_name = param_name_str

    param_noise_rate = df_perf_detector.loc[
        :, ["noise_rate", param_name_str, "success"]
    ]
    df_viz = param_noise_rate.groupby(["noise_rate", param_name_str]).mean()
    noise_rates = [x[0] for x in df_viz.index]
    param_values = [x[1] for x in df_viz.index]
    success = [x[0] for x in df_viz.values]
    df_viz = pd.DataFrame()
    for a, b, c in zip(noise_rates, param_values, success):
        df_viz.loc[a, b] = c  # print(a,b,c)
    df_viz = df_viz.loc[sorted(df_viz.index), sorted(df_viz.columns)]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[15, 4])
    sns.heatmap(data=df_viz.astype(float), ax=ax[0], cmap="Greens")
    ax[0].set_xlabel(beautiful_param_name)
    ax[0].set_ylabel("noise rate")
    ax[0].set_title(
        f"{detector_name} success depending on {beautiful_param_name}")

    param_noise_rate = df_perf_detector.loc[
        :, ["noise_rate", param_name_str, "performance", "success"]
    ]
    param_noise_rate.loc[:, "performance"] = param_noise_rate[
        ["success", "performance"]
    ].apply(lambda x: x[1] if x[0] else default_perf_if_fail, axis=1)
    param_noise_rate.loc[:, "performance"] = param_noise_rate.loc[
        :, "performance"
    ].apply(lambda x: x if x > 0 else default_perf_if_fail)
    param_noise_rate = param_noise_rate.drop(columns=["success"])
    df_viz = param_noise_rate.groupby(["noise_rate", param_name_str]).mean()
    noise_rates = [x[0] for x in df_viz.index]
    param_values = [x[1] for x in df_viz.index]
    success = [x[0] for x in df_viz.values]
    df_viz = pd.DataFrame()
    for a, b, c in zip(noise_rates, param_values, success):
        df_viz.loc[a, b] = c  # print(a,b,c)
    df_viz = df_viz.loc[sorted(df_viz.index), sorted(df_viz.columns)]

    sns.heatmap(data=df_viz.astype(float), ax=ax[1], cmap="Greens_r")
    ax[1].set_xlabel(beautiful_param_name)
    ax[1].set_ylabel("noise rate")
    ax[1].set_title(
        f"{detector_name} performance depending on {beautiful_param_name}")

    plt.savefig(
        f"/home/bastienzim/Documents/labs2/data/figures/{detector_name}_noise_delta.png"
    )
    return ax


def plot_detector_noise_scenario(
    df_perf_detector, param_name_str, detector_name, default_perf_if_fail
):
    if ("delta" in param_name_str):
        beautiful_param_name = r"$\delta$"
    elif ("alpha" in param_name_str):
        beautiful_param_name = r"$\alpha$"
    else:
        beautiful_param_name = param_name_str
    param_noise_rate = df_perf_detector.loc[
        :, ["noise_rate", "drift_type", "success"]
    ]
    df_viz = param_noise_rate.groupby(["noise_rate", "drift_type"]).mean()
    noise_rates = [x[0] for x in df_viz.index]
    param_values = [x[1] for x in df_viz.index]
    success = [x[0] for x in df_viz.values]
    df_viz = pd.DataFrame()
    for a, b, c in zip(noise_rates, param_values, success):
        df_viz.loc[b, a] = c  # print(a,b,c)
    df_viz = df_viz.loc[sorted(df_viz.index), sorted(df_viz.columns)]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[15, 4])
    sns.heatmap(data=df_viz.astype(float), ax=ax[0], cmap="Greens")
    ax[0].set_xlabel("noise rate")
    ax[0].set_ylabel("drift_type")
    ax[0].set_title(
        f"{detector_name} success depending on {beautiful_param_name}")

    param_noise_rate = df_perf_detector.loc[
        :, ["noise_rate", "drift_type", "performance", "success"]
    ]
    param_noise_rate.loc[:, "performance"] = param_noise_rate[
        ["success", "performance"]
    ].apply(lambda x: x[1] if x[0] else default_perf_if_fail, axis=1)
    param_noise_rate.loc[:, "performance"] = param_noise_rate.loc[
        :, "performance"
    ].apply(lambda x: x if x > 0 else default_perf_if_fail)
    param_noise_rate = param_noise_rate.drop(columns=["success"])
    df_viz = param_noise_rate.groupby(["noise_rate", "drift_type"]).mean()
    noise_rates = [x[0] for x in df_viz.index]
    param_values = [x[1] for x in df_viz.index]
    success = [x[0] for x in df_viz.values]
    df_viz = pd.DataFrame()
    for a, b, c in zip(noise_rates, param_values, success):
        df_viz.loc[b, a] = c  # print(a,b,c)
    df_viz = df_viz.loc[sorted(df_viz.index), sorted(df_viz.columns)]

    sns.heatmap(data=df_viz.astype(float), ax=ax[1], cmap="Greens_r")
    ax[1].set_xlabel("noise rate")
    ax[1].set_ylabel("drift_type")
    ax[1].set_title(
        f"{detector_name} performance depending on {beautiful_param_name}")

    plt.subplots_adjust(wspace=0.4)

    plt.savefig(
        f"/home/bastienzim/Documents/labs2/data/figures/{param_name_str}_noise_scenario.png"
    )
    return ax


def plot_detector_param_noise_scenario_specific(
    df_perf_detector,
    param_name_str,
    detector_name,
    default_perf_if_fail,
    selected_param_values,
):
    if ("delta" in param_name_str):
        beautiful_param_name = r"$\delta$"
    elif ("alpha" in param_name_str):
        beautiful_param_name = r"$\alpha$"
    else:
        beautiful_param_name = param_name_str

    param_noise_rate = df_perf_detector[
        df_perf_detector.loc[:, param_name_str].isin(selected_param_values)
    ].loc[:, ["noise_rate", "drift_type", "success"]]
    df_viz = param_noise_rate.groupby(["noise_rate", "drift_type"]).mean()
    noise_rates = [x[0] for x in df_viz.index]
    param_values = [x[1] for x in df_viz.index]
    success = [x[0] for x in df_viz.values]
    df_viz = pd.DataFrame()
    for a, b, c in zip(noise_rates, param_values, success):
        df_viz.loc[b, a] = c  # print(a,b,c)
    df_viz = df_viz.loc[sorted(df_viz.index), sorted(df_viz.columns)]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[15, 4])
    sns.heatmap(data=df_viz.astype(float), ax=ax[0], cmap="Greens")
    ax[0].set_xlabel("noise rate")
    ax[0].set_ylabel("drift_type")
    ax[0].set_title(
        f"{detector_name} success depending on {beautiful_param_name}")

    param_noise_rate = df_perf_detector[
        df_perf_detector.loc[:, param_name_str].isin(selected_param_values)
    ].loc[:, ["noise_rate", "drift_type", "performance", "success"]]
    param_noise_rate.loc[:, "performance"] = param_noise_rate[
        ["success", "performance"]
    ].apply(lambda x: x[1] if x[0] else default_perf_if_fail, axis=1)
    param_noise_rate.loc[:, "performance"] = param_noise_rate.loc[
        :, "performance"
    ].apply(lambda x: x if x > 0 else default_perf_if_fail)
    param_noise_rate = param_noise_rate.drop(columns=["success"])
    df_viz = param_noise_rate.groupby(["noise_rate", "drift_type"]).mean()
    noise_rates = [x[0] for x in df_viz.index]
    param_values = [x[1] for x in df_viz.index]
    success = [x[0] for x in df_viz.values]
    df_viz = pd.DataFrame()
    for a, b, c in zip(noise_rates, param_values, success):
        df_viz.loc[b, a] = c  # print(a,b,c)
    df_viz = df_viz.loc[sorted(df_viz.index), sorted(df_viz.columns)]

    sns.heatmap(data=df_viz.astype(float), ax=ax[1], cmap="Greens_r")
    ax[1].set_xlabel("noise rate")
    ax[1].set_ylabel("drift_type")
    ax[1].set_title(
        f"{detector_name} performance depending on {beautiful_param_name}")

    plt.subplots_adjust(wspace=0.4)
    plt.suptitle(beautiful_param_name+" = " +
                 str(selected_param_values[0]) + " " * 24)
    plt.savefig(
        f"/home/bastienzim/Documents/labs2/data/figures/{param_name_str}_noise_scenario_specific.png"
    )
    return ax


def compute_shap_diff(df_scenario, max_noise=0.25):
    """
    Shap - vanilla
    """
    df_res_diff = pd.DataFrame()
    for x in ["KSWIN", "PH", "adwin"]:
        selected_cols = sorted(
            [col for col in df_scenario.columns if x in col])[::-1]
        # print(f"i.e {selected_cols[0][8:]} is greater than {selected_cols[1][8:]}")
        df = (
            df_scenario[df_scenario.noise_rate <= max_noise]
            .loc[:, selected_cols + ["drift_type"]]
            .groupby("drift_type")
            .mean()
            .sort_values([x for x in selected_cols if "shap" in x])
        )
        df.loc[:, "prec_diff_" + x] = (
            df.loc[:, selected_cols[0]] - df.loc[:, selected_cols[1]]
        )  # .map("{:,.2f}".format)
        # print(df)
        # print("#"*100)
        df_res_diff = pd.concat(
            (df_res_diff, df.loc[:, ["prec_diff_" + x]]), axis=1)
    return (df_res_diff.astype(float))
