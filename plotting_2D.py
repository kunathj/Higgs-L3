import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import special
import stats as stat


#-------------------------------------------------------------------------------------------
def BkgSigHistos (backgrounds, signals, datas, variable_binning,x_label,savepath=None) :

    bkg = backgrounds
    sigModels = signals
    binning = variable_binning
    x_name, x_unit = x_label
    data_hist = datas


    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=(8,10))

    m_H = [85,90,95]

    for i in range(3) :
        binw = binning[i][1:]-binning[i][:-1]
        #binw = np.array([30,5,5,5,5,5,5,30])
        ax = axs[i]

        ax.bar(binning[i][:-1]+binw/2.,bkg[i],width=binw,label='background',color='yellow',edgecolor='k',linewidth=0.1)

        ax.bar(binning[i][:-1]+binw/2.,sigModels[i],width=binw,linewidth=0.1,
               label=r'H signal ($m_\mathrm{H}= $'+str(m_H[i])+' GeV)',color='red',bottom=bkg[i],edgecolor='k')

        # following http://ms.mcmaster.ca/peter/s743/poissonalpha.html
        # and https://newton.cx/~peter/2012/06/poisson-distribution-confidence-intervals/
        #lowerErr = data_hist[i] - np.array( special.gammaincinv(data_hist[i], 0.5*(1-0.6827)) )
        #lowerErr = [0 if entry != entry else entry for entry in lowerErr]
        #upperErr = np.array( special.gammaincinv(data_hist[i]+1, 1 - 0.5*(1-0.6827)) ) - data_hist[i]
        #datayErr = [lowerErr, upperErr] # Poissonian errors
        ax.errorbar(x=binning[i][:-1]+binw/2.,y=data_hist[i], xerr=binw/2., yerr=np.sqrt(data_hist[i]), #yerr=datayErr
                    fmt='o', color='k',label='data', linewidth=1)
        ax.legend(fontsize=14)
        ax.set_ylabel('Events / '+ str(round(binw[0],1))+' '+x_unit, fontsize=14)

    if (i == 2) :
        ax.set_xlabel(x_name+' ['+x_unit+']', fontsize=14)
        plt.tight_layout()

    fig.subplots_adjust(hspace=0)

    if (savepath != None) :
        #plt.savefig('plots/test')
        plt.savefig(savepath)
    plt.show()
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
def LogLikRatioPlots(arrays,obs,Nbins=30,savepath=None) :


    fig, axs = plt.subplots(nrows=3, ncols=1,figsize=(8,10))
    m_H = [85,90,95]
    QuantileList_b = []
    QuantileList_sPlusb = []

    CLlist = []
    for i in range(3) :
        ax = axs[i]

        llr_b, llr_sPlusb = arrays[i]
        norm = len(llr_b)
        binning = np.linspace(np.minimum(llr_b,llr_sPlusb).min(),np.maximum(llr_b,llr_sPlusb).max(),Nbins)

        #print(np.minimum(llr_b,llr_sPlusb).min())
        #print(norm)
        #print(binning)

        llr_b_hist = 1.*np.histogram(llr_b,bins=binning)[0]/norm
        QuantileList_b.append(stat.GetQuantiles(llr_b_hist,binning))
        if min(binning)<obs[i]:
            pos =  np.where(binning <= obs[i])[0][-1]
        else:
            pos = 0
            print("Higgs-Model %i: llr changed from %f to %f to fit into plot" %(m_H[i],obs[i],min(binning)))
        OneMinusCLb =  sum(llr_b_hist[:pos])
        llr_sPlusb_hist = 1.*np.histogram(llr_sPlusb,bins=binning)[0]/norm
        QuantileList_sPlusb.append(stat.GetQuantiles(llr_sPlusb_hist,binning))

        CLsPlusb =  sum(llr_sPlusb_hist[pos:])
        CLlist.append([OneMinusCLb, CLsPlusb])

        ax.step(x=binning[:-1],y=llr_b_hist,color='blue',label='bkg-like')
        ax.step(x=binning[:-1],y=llr_sPlusb_hist,color='red',label='sig+bkg-like')

        x1 = binning[binning<=obs[i]]
        x2 = binning[binning>obs[i]]
        width = binning[1]-binning[0]

        #ax.fill_between(x1,llr_b_hist[:len(x1)],color='red',alpha=0.5,interpolate=False)
        #ax.fill_between(x2,llr_sPlusb_hist[-len(x2):],color='blue',alpha=0.5,interpolate=True)
        ax.bar(x2[:-1]-width/2., llr_sPlusb_hist[-len(x2)+1:], width=width, color='blue', alpha=.5)
        ax.bar(x1-width/2., llr_b_hist[:len(x1)], width=width, color='red', alpha=.5)

        ax.set_xlabel(r'$-2 \ln (Q)$', fontsize=14)
        ax.set_ylabel('p.d.f.', fontsize=14)
        ax.set_title('signal model ' + r'($m_\mathrm{H} = $'+str(m_H[i])+' GeV)')
        ax.axvline(obs[i],label='observed',color='k')
        ax.legend(fontsize=14)

    plt.tight_layout()
    if (savepath != None) :
        #plt.savefig('plots/test')
        plt.savefig(savepath)
    plt.show()

    return CLlist, QuantileList_b, QuantileList_sPlusb
#-------------------------------------------------------------------------------------------


fs=12
#-------------------------------------------------------------------------------------------
def TwoDHist(var1, var2, framesMC_HiggsModels, NoHiggs, data, framesMC_HiggsModelsNames, savepath=None, bins=(40,40)) :

    # does not work for composed variable
    m_H = [85,90,95]
    for i,df in enumerate(framesMC_HiggsModels) :
        #dataframe = SelectionCut(dataframe=df) # remember to comment this in/out in both loops!
        if var2 == 'composed':
            var_2 = var2 + '_' + str(m_H[i])
        else: var_2 = var2
        dataframe = df
        plt.title (framesMC_HiggsModelsNames[i],fontsize=fs+2)
        plt.ylabel(var_2,fontsize=fs)
        plt.xlabel(var1,fontsize=fs)
        if len(dataframe) <= 1:
            print("Only %i events remaining in background %i after cuts!" %(len(dataframe), i))#framesMC_NoHiggsNames[i])
            continue
        plt.hist2d(dataframe[var1],
                        dataframe[var_2],
                        #bins = [var_dict[var1]['binning'],var_dict[var2]['binning']],
                        bins = bins,
                        weights=dataframe['weight'])
        plt.colorbar()
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath+var1+var_2+framesMC_HiggsModelsNames[i])
        plt.show()

    for j,df in enumerate([NoHiggs]):#(framesMC_NoHiggs) :
        #dataframe = SelectionCut(dataframe=df) # remember to comment this in/out in both loops!
        if var2 == 'composed':
            var_2 = var2 + '_' + str(m_H[j])
            print("Background and data distribution are printed for composed 85 GeV Higgs. For the other composed variables, choose them instead of composed as variable.")
        else: var_2 = var2
        dataframe = df
        plt.title ('background',fontsize=fs+2)
        plt.ylabel(var_2,fontsize=fs)
        plt.xlabel(var1,fontsize=fs)
        plt.hist2d(dataframe[var1],
                        dataframe[var_2],
                        bins=bins,
                        #bins = [var_dict[var1]['binning'],var_dict[var2]['binning']],
                        weights=dataframe['weight'])
        plt.colorbar()
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath+var1+var_2+'background')
        plt.show()

    for i,df in enumerate([data]):#(framesMC_NoHiggs) :
        #dataframe = SelectionCut(dataframe=df) # remember to comment this in/out in both loops!
        if var2 == 'composed':
            var_2 = var2 + '_' + str(m_H[i])
        else: var_2 = var2
        dataframe = df
        plt.title('data',fontsize=fs+2)
        plt.ylabel(var_2,fontsize=fs)
        plt.xlabel(var1,fontsize=fs)
        plt.hist2d(dataframe[var1],
                        dataframe[var_2],
                        bins=bins,
                        #bins = [var_dict[var1]['binning'],var_dict[var2]['binning']],
                        weights=dataframe['weight'])
        plt.colorbar()
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath+var1+var_2+'data')
        plt.show()
#-------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------
def TwoDHistFull(var1, var2, framesMC_HiggsModels, frames_NoHiggs, frames_data, framesMC_HiggsModelsNames, savepath=None, bins=(40,40)) :
    m_H = [85,90,95]
    for i,df in enumerate(framesMC_HiggsModels) :
        #dataframe = SelectionCut(dataframe=df) # remember to comment this in/out in both loops!
        if var2 == 'composed':
            var_2 = var2 + '_' + str(m_H[i])
        else: var_2 = var2
        dataframe = df
        plt.title (framesMC_HiggsModelsNames[i],fontsize=fs+2)
        plt.ylabel(var_2,fontsize=fs)
        plt.xlabel(var1,fontsize=fs)
        if len(dataframe) <= 1:
            print("Only %i events remaining in background %i after cuts!" %(len(dataframe), i))#framesMC_NoHiggsNames[i])
            continue
        plt.hist2d(dataframe[var1],
                        dataframe[var_2],
                        #bins = [var_dict[var1]['binning'],var_dict[var2]['binning']],
                        bins = bins,
                        weights=dataframe['weight'])
        plt.colorbar()
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath+var1+var_2+framesMC_HiggsModelsNames[i])
        plt.show()

        dataframe = frames_NoHiggs[i]
        plt.title ('background_'+str(m_H[i]),fontsize=fs+2)
        plt.ylabel(var_2,fontsize=fs)
        plt.xlabel(var1,fontsize=fs)
        plt.hist2d(dataframe[var1],
                        dataframe[var_2],
                        bins=bins,
                        #bins = [var_dict[var1]['binning'],var_dict[var2]['binning']],
                        weights=dataframe['weight'])
        plt.colorbar()
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath+var1+var_2+'background')
        plt.show()

        dataframe = frames_data[i]
        plt.title('data_'+str(m_H[i]),fontsize=fs+2)
        plt.ylabel(var_2,fontsize=fs)
        plt.xlabel(var1,fontsize=fs)
        plt.hist2d(dataframe[var1],
                        dataframe[var_2],
                        bins=bins,
                        #bins = [var_dict[var1]['binning'],var_dict[var2]['binning']],
                        weights=dataframe['weight'])
        plt.colorbar()
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath+var1+var_2+'data')
        plt.show()
#-------------------------------------------------------------------------------------------