import pandas as pd

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
