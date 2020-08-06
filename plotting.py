import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from sklearn.metrics import confusion_matrix

import variables

fontsize = 14

def BkgSigHistos(backgrounds, signals, datas, var, binning, asymm_errors=False):
    x_unit = variables.info[var]["x_unit"]
    if var in variables.info:
        x_name = variables.info[var]["x_name"]
    else:
        x_name = var

    x = (binning[1:] + binning[:-1]) /2.
    w = binning[1] - binning[0]

    fig, axs = plt.subplots(nrows=len(signals), ncols=1,
                            sharex=True, figsize=(8,10))
    for i, higgs_mass in enumerate(signals):
        ax = axs[i]
        ax.bar(x, backgrounds[higgs_mass], width=w, label="background",
               color="yellow", edgecolor="k", linewidth=0.1)
        ax.bar(x, signals[higgs_mass], width=w,
               label=f"H signal ($m_\mathrm{{H}}= ${higgs_mass[-2:]} GeV)",
               bottom=backgrounds[higgs_mass],
               color="red", edgecolor="k", linewidth=0.1)

        if asymm_errors:
            # Following http://ms.mcmaster.ca/peter/s743/poissonalpha.html
            # and https://newton.cx/~peter/2012/06/poisson-distribution-confidence-intervals/
            lower_err = (datas[higgs_mass]
                - special.gammaincinv(datas[higgs_mass], (1-0.6827)/2))
            lower_err = [0 if err != err else err for err in lower_err]
            upper_err = (-datas[higgs_mass]
                + special.gammaincinv(datas[higgs_mass]+1, 1-(1-0.6827)/2))
            data_y_err = [lower_err, upper_err]
        else:
            data_y_err = np.sqrt(datas[higgs_mass])  # Poissonian errors.
        ax.errorbar(x=x, y=datas[higgs_mass],
                    xerr=w/2., yerr=data_y_err,
                    label="data",
                    fmt="o", color="k", linewidth=1)
        ax.legend(fontsize=fontsize)
        ax.set_ylabel(f"Events / {round(w, 1)} {x_unit}", fontsize=fontsize)
        if (i+1 == len(axs)):
            ax.set_xlabel(f"{x_name} [{x_unit}]", fontsize=fontsize)
        plt.tight_layout()
    fig.subplots_adjust(hspace=0)


def getQuantiles(hist, binning):
    """Note that the histogram must be normalized.
    """
    cumulative = np.cumsum(hist)
    two_sigma_left =  binning[np.where(cumulative <=    .023)[0][-1]]
    one_sigma_left =  binning[np.where(cumulative <=    .160)[0][-1]]
    median =          binning[np.where(cumulative <=    .500)[0][-1]]
    one_sigma_right = binning[np.where(cumulative < 1 - .160)[0][-1]]
    two_sigma_right = binning[np.where(cumulative < 1 - .023)[0][-1]]
    return [median, [one_sigma_left, one_sigma_right],
                    [two_sigma_left, two_sigma_right]]

def LogLikRatioPlots(ratios, obs, n_bins=30):
    cl_s_and_b, quantiles_b, quantiles_s_plus_b  = {}, {}, {}

    fig, axs = plt.subplots(nrows=len(ratios), ncols=1, figsize=(8,10))
    for i, higgs_mass in enumerate(ratios):
        llr_b, llr_s_plus_b = ratios[higgs_mass]
        binning = np.linspace(np.minimum(llr_b, llr_s_plus_b).min(),
                              np.maximum(llr_b, llr_s_plus_b).max(), n_bins)
        width = binning[1] - binning[0]

        llr_b_hist = np.histogram(llr_b, bins=binning)[0] / len(llr_b)
        quantiles_b[higgs_mass] = getQuantiles(llr_b_hist, binning)

        llr_s_plus_b_hist = np.histogram(llr_s_plus_b, bins=binning)[0] / len(llr_b)
        quantiles_s_plus_b[higgs_mass] = getQuantiles(llr_s_plus_b_hist, binning)

        axs[i].step(x=binning[:-1],y=llr_b_hist,
                    color="blue",label="bkg-like")
        axs[i].step(x=binning[:-1],y=llr_s_plus_b_hist,
                    color="red",label="sig+bkg-like")

        x1 = binning[binning <= obs[higgs_mass]]
        x2 = binning[binning >  obs[higgs_mass]]
        axs[i].bar(x2[:-1]-width/2., llr_s_plus_b_hist[-len(x2)+1:],
                   width=width, color="blue", alpha=.5)
        axs[i].bar(x1-width/2., llr_b_hist[:len(x1)],
                   width=width, color="red", alpha=.5)

        axs[i].set_xlabel("$-2 \ln (Q)$", fontsize=fontsize)
        axs[i].set_ylabel("p.d.f.", fontsize=fontsize)
        axs[i].set_title(f"signal model ($m_\mathrm{{H}} = $ {higgs_mass[-2:]} "
                         "GeV)")
        axs[i].axvline(obs[higgs_mass], label="observed", color="k")
        axs[i].legend(fontsize=fontsize)

        if min(binning) < obs[higgs_mass]:
            obs_bin =  np.where(binning <= obs[higgs_mass])[0][-1]
        else:
            obs_bin = min(binning)
            print(f"Higgs-Model {higgs_mass}: Observed llr changed from "
                f"{obs[higgs_mass]} to {obs_bin} to fit into plot.")
        one_minus_cl_b =  sum(llr_b_hist[:obs_bin])
        cl_s_plus_b =  sum(llr_s_plus_b_hist[obs_bin:])
        cl_s_and_b[higgs_mass] = [one_minus_cl_b, cl_s_plus_b]
    plt.tight_layout()
    return cl_s_and_b, quantiles_b, quantiles_s_plus_b


def confusionMatrix(y_pred, y_test):
    cm_total_counts = confusion_matrix(y_pred,y_test)
    cm_normalized = cm_total_counts / cm_total_counts.sum(axis=0)
    cm = cm_normalized.T

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    thresh = cm.min() + (cm.max() - cm.min())/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{np.round(cm[i, j]*100, 0)} %",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    classes = ["bkg", "sig"]
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel("predicted category")
    plt.ylabel("true category")