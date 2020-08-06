import itertools
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from .helpers import addLogisticRegressionResults, bdt_cut, kinematical_vars


luminosity = 176.773
data = pd.read_csv("data/higgs_data.csv")
data["weight"] = 1

cross_sections_higgs_models = {
    "higgs_85": 0.0094,
    "higgs_90": 0.0667,
    "higgs_95": 0.0333,
}
n_mc_higgs_models = {
    "higgs_85": 3972,
    "higgs_90": 3973,
    "higgs_95": 3971,
}
mc_higgs_models = {}
for mass in n_mc_higgs_models:
    mc_higgs_models[mass] = pd.read_csv(f"data/higgs_{mass}.csv")
    mc_higgs_models[mass]["weight"] = luminosity * (
        cross_sections_higgs_models[mass] / n_mc_higgs_models[mass])
    mc_higgs_models[mass]["class"] = 1

cross_sections_no_higgs = {
    "eeqq": 15600,
    "qq": 102,
    "wen": 2.9,
    "ww": 16.5,
    "zee": 3.35,
    "zz": 0.975,
}
n_mc_no_higgs = {
    "eeqq": 5940000,
    "qq": 200000,
    "wen": 81786,
    "ww": 294500,
    "zee": 29500,
    "zz": 196000,
}
mc_no_higgs_frames = {}
for process in n_mc_no_higgs:
    mc_no_higgs_frames[process] = pd.read_csv(f"data/higgs_{process}.csv")
    mc_no_higgs_frames[process]["weight"] = luminosity * (
        cross_sections_no_higgs[process] / n_mc_no_higgs[process])
    mc_no_higgs_frames[process]["class"] = 0


def getTrainAndTest(higgs_mass, bdt_precut=False, drop=None,
                    upweight_signal=False
):
    df_no_higgs = pd.concat(mc_no_higgs_frames)
    df_higgs = mc_higgs_models[higgs_mass]
    df_MVA = pd.concat([df_no_higgs, df_higgs])
    df_MVA = df_MVA[kinematical_vars + ["class", "weight"]]

    if bdt_precut:
        with open(f"tmp/BDT_{higgs_mass}.pkl", "rb") as fid:
            bdt_loaded = pickle.load(fid)
        df_MVA["bdt"] = bdt_loaded.decision_function(df_MVA[kinematical_vars])
        df_MVA = df_MVA[df_MVA["bdt"] > bdt_cut[higgs_mass]]
        del df_MVA["bdt"]

    if drop is not None:
        df_MVA.drop(columns=drop)

    if upweight_signal:
        sig_weight = sum(df_MVA[df_MVA["class"] == 1]["weight"])
        bkg_weight = sum(df_MVA["weight"]) - sig_weight
        df_MVA["weight"][df_MVA["class"] == 1] *= bkg_weight / sig_weight
    target = df_MVA.pop("class")

    X_train, X_test, y_train, y_test = train_test_split(df_MVA, target,
        stratify=target, random_state=42, train_size=0.65)

    X_train_w = X_train.pop("weight")
    X_test_w = X_test.pop("weight")

    return X_train, X_test, y_train, y_test, X_train_w, X_test_w


def getPreselectedSBD():
    bdt_collection = {}
    for higgs_mass in mc_higgs_models:
        with open(f"tmp/BDT_{higgs_mass}.pkl", "rb") as fid:
            bdt_collection[higgs_mass] = pickle.load(fid)

    for frame in itertools.chain(mc_higgs_models.values(),
                                mc_no_higgs_frames.values(),
                                [data]):
        addLogisticRegressionResults(df=frame)
        for higgs_mass in mc_higgs_models:
            bdt_response = bdt_collection[higgs_mass].decision_function
            frame[f"BDT_selCut{higgs_mass[-2:]}"] = bdt_response(
                frame[kinematical_vars])

    vars = kinematical_vars + ["weight"]
    sig, bkg, dat = {}, {}, {}
    for m_h in mc_higgs_models:
        s = mc_higgs_models[m_h]
        b = pd.concat(mc_no_higgs_frames, ignore_index=True)
        d = data
        sig[m_h] = s[s[f"BDT_selCut{m_h[-2:]}"] > bdt_cut[m_h]][vars]
        bkg[m_h] = b[b[f"BDT_selCut{m_h[-2:]}"] > bdt_cut[m_h]][vars]
        dat[m_h] = d[d[f"BDT_selCut{m_h[-2:]}"] > bdt_cut[m_h]][vars]
    return sig, bkg, dat
