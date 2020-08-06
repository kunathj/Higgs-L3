import numpy as np
import matplotlib.pyplot as plt
import os
import stats as stat

from plotting import *


fs=12
def TwoDHist(var1, var2, mc_higgs_models, mc_no_higgs, data, savepath=None, bins=(40,40)) :

    # does not work for composed variable
    for m_higgs_name, df in mc_higgs_models.items():
        if var2 == 'composed':
            var_2 = var2 + '_' + str(m_higgs_name)
        else: var_2 = var2
        plt.title(m_higgs_name, fontsize=fs+2)
        plt.ylabel(var_2,fontsize=fs)
        plt.xlabel(var1,fontsize=fs)
        if len(df) <= 1:
            print("Only %i events remaining in background %i after cuts!" %(len(df), i))
            continue
        plt.hist2d(df[var1], df[var_2], bins = bins, weights=df["weight"])
        plt.colorbar()
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath+var1+var_2+m_higgs_name)
        plt.show()

    if var2 == 'composed':
        var_2 = var2 + "_higgs_85"
        print("Background and data distribution are printed for composed 85 GeV Higgs. For the other composed variables, choose them instead of composed as variable.")
    else:
        var_2 = var2
    df = mc_no_higgs
    plt.title ('background',fontsize=fs+2)
    plt.ylabel(var_2,fontsize=fs)
    plt.xlabel(var1,fontsize=fs)
    plt.hist2d(df[var1], df[var_2], bins=bins, weights=df["weight"])
    plt.colorbar()
    plt.tight_layout()
    if savepath != None:
        plt.savefig(savepath+var1+var_2+'background')
    plt.show()

    df  = data
    plt.title('data',fontsize=fs+2)
    plt.ylabel(var_2,fontsize=fs)
    plt.xlabel(var1,fontsize=fs)
    plt.hist2d(df[var1], df[var_2], bins=bins, weights=df["weight"])
    plt.colorbar()
    plt.tight_layout()
    if savepath != None:
        plt.savefig(savepath+var1+var_2+'data')
    plt.show()