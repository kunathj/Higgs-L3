import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import special, stats
import stats as stat
import variables

fontsize = 14

def BkgSigHistos(backgrounds, signals, datas, var, binning,
                 asymm_errors=False, save_as=None,
):
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
    if save_as != None:
        plt.savefig(save_as)




def LogLikRatioPlots(arrays,obs,Nbins=30,savepath=None) :


    fig, axs = plt.subplots(nrows=3, ncols=1,figsize=(8,10))
    m_H = [85,90,95]

    CLlist = []

    QuantileList_b = []
    QuantileList_sPlusb = []


    for i in range(3) :
        ax = axs[i]

        llr_b, llr_sPlusb = arrays[i]
        norm = len(llr_b)
        binning = np.linspace(np.minimum(llr_b,llr_sPlusb).min(),np.maximum(llr_b,llr_sPlusb).max(),Nbins)
        width = binning[1]-binning[0]


        llr_b_hist = 1.*np.histogram(llr_b,bins=binning)[0]/norm
        QuantileList_b.append(stat.GetQuantiles(llr_b_hist,binning))


        pos =  np.where(binning <= obs[i])[0][-1]

        if min(binning)<obs[i]:
            pos =  np.where(binning <= obs[i])[0][-1]
        else:
            pos = 0
            print("Higgs-Model %i: llr changed from %f to %f to fit into plot" %(m_H[i],obs[i],min(binning)))
        print(pos)
        OneMinusCLb =  sum(llr_b_hist[:pos])

        llr_sPlusb_hist = 1.*np.histogram(llr_sPlusb,bins=binning)[0]/norm
        QuantileList_sPlusb.append(stat.GetQuantiles(llr_sPlusb_hist,binning))

        CLsPlusb =  sum(llr_sPlusb_hist[pos:])
        CLlist.append([OneMinusCLb, CLsPlusb])

        ax.step(x=binning[:-1],y=llr_b_hist,color='blue',label='bkg-like')
        ax.step(x=binning[:-1],y=llr_sPlusb_hist,color='red',label='sig+bkg-like')

        x1 = binning[binning<=obs[i]]
        x2 = binning[binning>obs[i]]

        #ax.fill_between(x1,llr_b_hist[:len(x1)],color='red',alpha=0.5,interpolate=True)
        #ax.fill_between(x2,llr_sPlusb_hist[-len(x2):],color='blue',alpha=0.5,interpolate=True)

        ax.bar(x2[:-1]-width/2., llr_sPlusb_hist[-len(x2)+1:], width=width, color='blue', alpha=.5)
        ax.bar(x1-width/2., llr_b_hist[:len(x1)], width=width, color='red', alpha=.5)


        ax.set_xlabel(r'$-2 \ln (Q)$',fontsize=14)
        ax.set_ylabel('p.d.f.',fontsize=14)
        ax.set_title('signal model ' + r'($m_\mathrm{H} = $'+str(m_H[i])+' GeV)')
        ax.axvline(obs[i],label='observed',color='k')
        ax.legend(fontsize=14)

    plt.tight_layout()
    if (savepath != None) :
        #plt.savefig('plots/test')
        plt.savefig(savepath)
    else :
        plt.show()



    return CLlist, QuantileList_b, QuantileList_sPlusb