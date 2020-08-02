import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import stats as stat

#-------------------------------------------------------------------------------------------
def BkgSigHistos (background, signals, data, variable_binning,x_label,savepath=None) :

    bkg = background
    sigModels = signals
    binning = variable_binning
    x_name, x_unit = x_label
    data_hist = data

    binw = binning[1]-binning[0]
    #binw = np.array([10,10,10,5,5,5,5,5,5,10,20])
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=(8,10))

    m_H = [85,90,95]

    for i in range(3) :

        ax = axs[i]

        ax.bar(binning[:-1]+binw/2.,bkg[i],width=binw,label='background',color='yellow',edgecolor='k',linewidth=0.1)

        ax.bar(binning[:-1]+binw/2.,sigModels[i],width=binw,linewidth=0.1,
               label=r'H signal ($m_\mathrm{H}= $'+str(m_H[i])+' GeV)',color='red',bottom=bkg[i],edgecolor='k')

        ax.errorbar(x=binning[:-1]+binw/2.,y=data_hist[i], xerr=binw/2., yerr=np.sqrt(data_hist[i]),
                    fmt='o', color='k',label='data',linewidth=1)
        ax.legend(fontsize=14)
        ax.set_ylabel('Events / '+ str(round(binw,1))+' '+x_unit,fontsize=14)
        if (i == 2) :
            ax.set_xlabel(x_name+' ['+x_unit+']',fontsize=14)
        plt.tight_layout()

    fig.subplots_adjust(hspace=0)

    if (savepath != None) :
        #plt.savefig('plots/test')
        plt.savefig(savepath)
    else :
        plt.show()
#-------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------------