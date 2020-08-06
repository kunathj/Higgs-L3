from collections import defaultdict
import numpy as np

info = defaultdict(lambda: dict(
    x_name="default",
    x_unit="$1$",
    binning=None,
))

info["mmis"] = dict(
    x_name="missing mass $m_\mathrm{mis}$",
    x_unit="$\mathrm{GeV} / \mathrm{c}^2$",
    binning=np.linspace(50, 130, 28),
)

info["acop"] = dict(
    x_name="$\pi -$ angles btw jets",
    x_unit="$\mathrm{rad}$",
    binning=np.linspace(2.9, np.pi, 30),
)

info["acthm"] = dict(
    x_name="$|\cos(\phi_\mathrm{polar,pmiss})|$",
    x_unit="$1$",
    binning=np.linspace(0, 1, 20),
)

info["fmvis"] = dict(
    x_name="visible mass (Z-adjusted) $m_\mathrm{vis}$",
    x_unit="GeV",
    binning=np.append(np.linspace(60, 100, 20)[:-5], 100),
)

info["mvis"] = dict(
    x_name="visible mass $m_\mathrm{vis}$",
    x_unit="GeV",
    binning=np.linspace(70, 100, 15),
)

info["xmj1"] = dict(
    x_name="more energetic 2-jet event mass $m_\mathrm{jet1}$",
    x_unit="GeV",
    binning=np.linspace(5, 20, 12),
)

info["xmj2"] = dict(
    x_name="less energetic 2-jet event mass $m_\mathrm{jet2}$",
    x_unit="GeV",
    binning=np.linspace(5, 20, 5),
)

info["ucsdbt0"] = dict(
    x_name="B tag, based on tracking information only",
    x_unit="$1$",
    binning=np.append(np.linspace(0, 8, 10), np.array([10, 14])),
)

info["btag1"] = dict(
    x_name="more energetic B tag",
    x_unit="$1$",
    binning=np.linspace(0, 1, 30),
)

info["btag2"] = dict(
    x_name="less energetic B tag",
    x_unit="$1$",
    binning=np.linspace(0, 1, 30),
)

info["ievt"] = dict(
    x_name="event number",
    x_unit="$1$",
    binning=np.linspace(0, 22000, 30),
)

info["maxxov"] = dict(
    x_name="maxxov",
    x_unit="$1$",
    binning=np.linspace(0,1,20),
)

info["composed"] = dict(
    x_name="composition of multiple variables",
    x_unit="$1$",
    binning=np.linspace(-14, 8, 11),
)