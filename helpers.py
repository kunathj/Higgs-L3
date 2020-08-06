import numexpr
import numpy as np
from pathlib import Path

higgs_hypothesis = ["higgs_85", "higgs_90", "higgs_95"]
# Choose only those columns which have some kinematical meaning
# and thus can be used for training.
kinematical_vars = [
    "btag1", "btag2",
    "ucsdbt0", "mvis", "mvissc", "fmvis", "fmmis", "fth1",
    "mmis", "acthm", "maxcthj", "acop", "maxxov", "enj1",
    "thj1", "phj1", "xmj1", "enj2", "thj2", "phj2", "xmj2",
    "pho_num", "pho_ene", "pho_the", "pho_phi", "ele_num", "ele_ene",
    "ele_the", "ele_phi", "muon_num", "muon_ene", "muon_the",
    "muon_phi"
]

variable_dists_dir = Path("plots/VariableDists")
log_likeliratio_dir = Path("plots/loglikeliratio")
log_reg_coeffs_dir = Path("tmp/log_reg_coeffs")

def byEyeSelectionCut(df, m_higgs_hypothesis=85, out=False):
    bvalue = 0.18
    cut_exprs = [
        f"(btag1 > {bvalue}) | (btag1 > {bvalue})",
        f"mmis > {65}",
        f"mvis < {m_higgs_hypothesis+5}",
        f"fmvis < {m_higgs_hypothesis+5}",
        f"ucsdbt0 > {1.4}",
    ]
    if out == True:
        mask = np.full(len(df), True, dtype=bool)
        print("No selection:", sum(mask))
        for cut_expr in cut_exprs:
            add_on_mask = numexpr.evaluate(cut_expr, df)
            mask = mask & add_on_mask
            print(sum(mask), "for", cut_expr)
    else:
        cut_expr = "(" + ") & (".join(cut_exprs) + ")"
        mask = numexpr.evaluate(cut_expr, df)
    return mask

def addLogisticRegressionResults(df):
    coeffs = {}
    variables = [var for var in kinematical_vars if var != "fmmis"]
    for m_h in higgs_hypothesis:
        coeffs[m_h] = np.loadtxt(log_reg_coeffs_dir / f"{m_h}.txt")
        name = "composed_" + m_h
        counts = 0
        for i, var in enumerate(variables):
            counts += coeffs[m_h][i] * df[var]
        counts += coeffs[m_h][-1]  # The intercept.
        df[name] = counts

bdt_cut = {
    "higgs_85": -3.15,
    "higgs_90": -2.9,
    "higgs_95": -2.9,
}

symbol = dict(higgs_85="r^", higgs_90="b*", higgs_95="go")