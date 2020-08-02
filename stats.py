import numpy as np


#-------------------------------------------------------------------------------------------
def LogLikRatio (background, signal, N_experiments=10000) :


    b = background
    s = signal
    s_tot = s.sum()

    llr_b_like = []
    llr_sPlusb_like = []

    for k in range(N_experiments) :
        N_b = np.random.poisson(lam=b)
        N_sPlusb = np.random.poisson(lam=(s+b))

        llr_b_like.append(2*s_tot - 2*np.dot(N_b,np.log(1+s/b)))
        llr_sPlusb_like.append(2*s_tot - 2*np.dot(N_sPlusb,np.log(1+s/b)))

    return llr_b_like, llr_sPlusb_like
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
def LogLikRatioObserved (background, signals, data) :


    b = background
    N = data
    llr_data_is_b_like = []
    for s in signals :
        s_tot = s.sum()

        llr_data_is_b_like.append(2*s_tot - 2*np.dot(N,np.log(1+s/b)))


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