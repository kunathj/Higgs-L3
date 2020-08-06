import numpy as np


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


def LogLikRatioObserved(background, signal, data):
    s, b, toy_llr = prepareLikelihoodRatio(signal, background)
    d = data[np.where(background != 0)]
    return toy_llr(d)