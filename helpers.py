import numexpr
import numpy as np
import os

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

variable_dists_dir = "plots/VariableDists"
log_likeliratio_dir = "plots/loglikeliratio"

for directory in [variable_dists_dir, log_likeliratio_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)


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


def addComposedVariable(df, with_mmis=False):
    coeffs = {}
    if with_mmis:
        variables = kinematical_vars
        coeffs["85"] = [
            2.151, 2.227, 0.06, -0.341, -0.23,
            0.188, -1.707, 0.518, 0.821, -1.254,
            -1.766, -2.097, 1.548, 1.331, -0.528,
            0.03, -0.056, 1.256, -0.124, -0.032,
            -0.068, 0.139, -0.058, 0.092, -0.015,
            -0.315, -0.002, 0.05, 0.057, -0.316,
            -0.007, -0.314, 0.099
        ]
        coeffs["90"] = coeffs["85"]
        coeffs["95"] = coeffs["85"]
    else:
        variables = [var for var in kinematical_vars if var != "fmmis"]
        coeffs["85"] = [
            0.313, 0.195, 0.284, -0.002, -0.031,
            0.069, -0.019, -0.053, -0.173, -0.232,
            -0.295, 0.011, 0.004, -0.019, 0.009,
            -0.052, -0.033, -0.055, -0.003, -0.036,
            0.106, -0.059, -0.006, 0.034, 0.011,
            -0.027, -0.032, 0.057, 0.001, -0.003,
            -0.011, 0.024
        ]
        coeffs["90 old"] = [
            0.127, 0.184, 0.217, -0.043, -0.029,
            -0.016, 0.044, -0.046, -0.084, -0.149,
            -0.297, 0.002, 0.044, 0.051, 0.011,
            -0.049, 0.039, 0.029, 0.012, -0.03,
            0.072, -0.036, 0.03 , -0.001, 0.053,
            -0.03 , -0.026, 0.028, -0.018, -0.004,
            -0.032, -0.046
        ]
        coeffs["90"] = [
            1.665, 1.778, 0.019, -0.102, -0.014,
            0.03 , 0.016, -0.335, 0.243, -1.119,
            -1.394, 0.482, 0.099, 0.502, 0.01,
            -0.065, 0.074, 0.187, 0.017, -0.042,
            0.209, -0.039, 0.031, -0.024, 0.438,
            -0.039, -0.255, 0.014, -0.267, -0.002,
            -0.056, -0.02
        ]
        coeffs["90 new"] = [
            0.158, 0.244, 0.335, -0.027, -0.078,
            -0.033, 0.115, -0.09 ,-0.047, -0.152,
            -0.198, -0.002, -0., -0.017, -0.01,
            -0.023, 0.035, -0.069, -0.048, -0.042,
            0.11 , -0.004, 0.012, 0.022, 0.009,
            -0.032, -0.026, -0.002, -0.024, -0.,
            -0.02 , -0.14
        ]
        coeffs["95"] = [
            0.106, 0.048, 0.134, -0.091, -0.146,
            0.059, 0.116, 0.008,-0.073, -0.094,
            -0.192, -0.001, 0.059, 0.021, 0.062,
            -0.023, 0.039, 0.059, 0.018, -0.012,
            0.035, -0.043, 0.069, 0.016,-0.002,
            -0.022, 0.001, 0.057, 0.006, 0.005,
            0.013, 0.027
        ]
    for mass in ["85", "90", "95"]:
        name = "composed_" + mass
        if with_mmis:
            name += "_with_mmis"
        for i, var in enumerate(variables):
            count += coeffs[mass][i] * df[var]
        df[name] = counts
