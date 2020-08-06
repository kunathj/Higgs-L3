import numpy as np


#-------------------------------------------------------------------------------------------
def prepareLikelihoodRatio(signal, background):
    if sum(signal[np.where(background == 0)]) != 0:
        print("Weird behavior: We found bins with zero bkg expectation and "
              "non-zero signal expectation. This should not happen. "
              "The bins are skipped.")
    b = background[np.where(background != 0)]
    s = signal[np.where(background != 0)]
    def toy_llr(n_random):
        return 2*s.sum() - 2*np.dot(n_random, np.log(1 + s/b))
    return s, b, toy_llr


def LogLikRatio(background, signal, n_experiments=10000):
    s, b, toy_llr = prepareLikelihoodRatio(signal, background)

    llr_b_like = []
    llr_s_plus_b_like = []
    for k in range(n_experiments):
        n_b = np.random.poisson(lam=b)
        n_s_plus_b = np.random.poisson(lam=(s+b))

        llr_b_like.append(toy_llr(n_b))
        llr_s_plus_b_like.append(toy_llr(n_s_plus_b))
    return llr_b_like, llr_s_plus_b_like
#-------------------------------------------------------------------------------------------


def LogLikRatioObserved(background, signal, data):
    s, b, toy_llr = prepareLikelihoodRatio(signal, background)
    d = data[np.where(background != 0)]
    return toy_llr(d)


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

def GetCLOneMinusb(array, cut):
    hist, binning = np.histogram(array,bins=300)
    hist = hist / (1.*len(array))


    OneMinusCLb = []
    for c in cut :
        try :
            pos =  np.where(binning <= c)[0][-1]
            OneMinusCLb.append(sum(hist[:pos]))
        except :
            OneMinusCLb.append(0)

    return OneMinusCLb