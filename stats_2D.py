import numpy as np

#-------------------------------------------------------------------------------------------
def LogLikRatio (background, signal, N_experiments=10000) :


    b = background
    s = signal
    s_tot = s.sum()

    llr_b_like = []
    llr_sPlusb_like = []

    N_emptyBins = 0

    for k in range(N_experiments) :
        N_b = np.random.poisson(lam=b)
        N_sPlusb = np.random.poisson(lam=(s+b))

        b_like, sPlusb_like = [0,0]
        for i in range(len(b)) :
            if b[i] == 0 :
                if s[i] != 0 :
                    b_like += 2*s[i]
                    if N_b[i] != 0 :
                        print("There was a bin with %f signal and %f data counts, but no background. As the ratio would blow up due to division by zero, we did not include it." %(s[i], N_b[i]))
                    if N_sPlusb[i] != 0:
                        pass#b_like += -np.inf
                else:
                    N_emptyBins += 1
            else:
                b_like += 2*s[i] - 2*N_b[i]*np.log(1+s[i]/b[i])
                sPlusb_like += 2*s[i] - 2*N_sPlusb[i]*np.log(1+s[i]/b[i])
        llr_b_like.append(b_like)
        llr_sPlusb_like.append(sPlusb_like)

    return llr_b_like, llr_sPlusb_like
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
"""
In case there are different backgrounds due to different cuts per signal hypothesis applied
"""
def LogLikRatiosObserved (backgrounds, signals, datas) :

    N_emptyBins = 0
    llr_data_is_b_like = []
    for l,s in enumerate(signals) :
        b = backgrounds[l]
        N = datas[l]
        b_like = 0
        for i in range(len(b)) :
            if b[i] == 0 :
                if s[i] != 0 :
                    b_like += 2*s[i]
                    if N[i] != 0 :
                        print("There was a bin with %f signal and %f data counts, but no background. As the ratio would blow up due to division by zero, we did not include it." %(s[i], N[i]))
                        pass#b_like += -np.inf
                else:
                    N_emptyBins += 1
            else:
                b_like += 2*s[i] - 2*N[i]*np.log(1+s[i]/b[i])
        llr_data_is_b_like.append(b_like)
    return llr_data_is_b_like
#-------------------------------------------------------------------------------------------

"""
2D
"""

#-------------------------------------------------------------------------------------------
def LogLikRatio_TwoD (background, signal, N_experiments=10000, emptyCalc = False) :


    b = background
    s = signal

    llr_b_like = []
    llr_sPlusb_like = []

    N_emptyBins = 0

    for k in range(N_experiments) :
        N_b = np.random.poisson(lam=b)
        N_sPlusb = np.random.poisson(lam=(s+b))

        # Here, we have to take care of empty bins (can happen in the 2D case)
        # (As in 1D, where the bins just start at the first populated place, we ignore empty bins)
        b_like, sPlusb_like = [0,0]
        for i in range(len(b)):
            for j in range(len(b[0])):
                if b[i,j] == 0:
                    if s[i,j] != 0 :
                        b_like += 2*s[i,j]
                        if N_b[i,j] != 0 :
                            print("There was a bin with %f signal and %f data counts, but no background. As the ratio would blow up due to division by zero, we did not include it." %(s[i,j], N_b[i,j]))
                        if N_sPlusb[i,j] != 0:
                            pass#b_like += -np.inf
                        else: N_emptyBins += 1
                else:
                        b_like += 2*s[i,j] - 2*N_b[i,j]*np.log(1+s[i,j]/b[i,j])
                        sPlusb_like += 2*s[i,j] - 2*N_sPlusb[i,j]*np.log(1+s[i,j]/b[i,j])
        llr_b_like.append(b_like)
        llr_sPlusb_like.append(sPlusb_like)
    if emptyCalc == True:
        print("%i out of %i bins where empty." %(N_emptyBins, N_experiments*len(b)*len(b[0])))
    return llr_b_like, llr_sPlusb_like
#-------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------
def LogLikRatioObserved_TwoD (bkgModels, signals, data_histModels, emptyCalc = False) :

    llr_data_is_b_like = []
    N_emptyBins = 0
    for k,s in enumerate(signals) :
        b = bkgModels[k]
        N = data_histModels[k]
        # Here, we have to take care of empty bins (can happen in the 2D case)
        # (As in 1D, where the bins just start at the first populated place, we ignore empty bins)
        data_is_b_like = 0
        for i in range(len(b)):
            for j in range(len(b[0])):
                if b[i,j] == 0:
                    if s[i,j] != 0 :
                        data_is_b_like += 2*s[i,j]
                        if N[i,j] != 0 :
                            print("There was a bin with %f signal and %f data counts, but no background. As the ratio would blow up due to division by zero, we did not include it." %(s[i,j], N[i,j]))
                        else: N_emptyBins += 1
                else:
                        data_is_b_like += 2*s[i,j] - 2*N[i,j]*np.log(1+s[i,j]/b[i,j])
        llr_data_is_b_like.append(data_is_b_like)
    if emptyCalc == True:
        print("%i out of %i bins where empty." %(N_emptyBins, len(signals)*len(b)*len(b[0])))

    return llr_data_is_b_like
#-------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------
def GetQuantiles (hist,binning) :
    # note that the histogram must be normalized
    cumulative = np.cumsum(hist)

    twoSigmaLeft = 0.023
    oneSigmaLeft = 0.16
    median = 0.5
    oneSigmaRight = 1.-0.16
    twoSigmaRight = 1.-0.023


    TwoSigmaLeft = binning[np.where(cumulative <= twoSigmaLeft)[0][-1]]
    OneSigmaLeft = binning[np.where(cumulative <= oneSigmaLeft)[0][-1]]
    Median = binning[np.where(cumulative <= median)[0][-1]]
    OneSigmaRight = binning[np.where(cumulative < oneSigmaRight)[0][-1]]
    TwoSigmaRight = binning[np.where(cumulative < twoSigmaRight)[0][-1]]


    return [Median,[OneSigmaLeft,OneSigmaRight],[TwoSigmaLeft,TwoSigmaRight]]
#-------------------------------------------------------------------------------------------