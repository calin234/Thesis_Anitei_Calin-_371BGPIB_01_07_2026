import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

#!/usr/bin/env python3
"""
=============================================================================
  MOSFET SPICE Level 1 Parameter Extractor  —  multi-curve version
  -----------------------------------------------------------------
  Extracts VTO, KP (effective), LAMBDA, and RDS(on) from output
  characteristics I_D vs V_DS at any number of fixed V_GS values.

  Input CSV structure
  -------------------
  Row 0  : header with V_GS values, e.g.:
              "Gate Voltage, 4.0, Gate Voltage, 5.0, ..., Gate Voltage, 9.0"
  Rows 1+: pairs of columns  VDS[V] | ID[A or mA]  for each V_GS value

  The script auto-detects V_GS values from the header (the numeric tokens
  at positions 1, 3, 5, … of the header row) and handles any number of
  V_GS curves.

  SPICE Level 1 (Shichman-Hodges) equations
  ------------------------------------------
  Saturation  (V_DS >= V_GS - VTO):
    I_D = (KP/2) * (W/L) * (V_GS - VTO)^2 * (1 + LAMBDA * V_DS)

  Linear  (0 <= V_DS < V_GS - VTO):
    I_D = KP * (W/L) * [(V_GS-VTO)*V_DS - V_DS^2/2] * (1 + LAMBDA*V_DS)

  Note on KP
  ----------
  KP_eff = KP_intrinsic * (W/L) is extracted (W and L are unknown).
  Set  W=1  L=1  on every MOSFET instance in the netlist.

  Dependencies
  ------------
  pip install numpy pandas matplotlib scipy
=============================================================================
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =============================================================================
#  ▶  USER SETTINGS  — edit these before running
# =============================================================================

CSV_FILE    = "final15_06.csv"     # ← your CSV filename
DEVICE_NAME = "IRF740"

ID_UNIT     = "A"                  # "A" or "mA"
CSV_SEP     = None                 # None = auto-detect ("," or ";")
CSV_DECIMAL = "."                  # "." standard  or "," for Romanian locale

# Saturation region detection
SAT_FRAC    = 0.85    # points with I_D >= SAT_FRAC * max(I_D) → saturation
MIN_SAT_PTS = 3       # minimum saturation points for a valid fit

# RDS(on) linear-region bound (fraction of max V_DS for the top V_GS curve)
LIN_FRAC    = 0.10

# Knee detection threshold (for VTO fallback)
KNEE_FRAC   = 0.85

# VTO validity range [V] — must lie in this window to be physical
VTO_MIN, VTO_MAX = 0.5, 5.0

# Self-heating flag: curve is "distorted" when its lambda exceeds
# LAMBDA_RATIO_WARN times the minimum lambda across all curves
LAMBDA_RATIO_WARN = 5.0

PLOT_FILE   = f"{DEVICE_NAME}_SPICE_extraction.png"

# =============================================================================
# 1.  LOAD DATA  — auto-detect V_GS values and number of curves
# =============================================================================

def load_data():
    print(f"\n  Reading '{CSV_FILE}' …")
    try:
        df = pd.read_csv(
            CSV_FILE, header=0,
            sep=CSV_SEP,
            engine="python" if CSV_SEP is None else "c",
            decimal=CSV_DECIMAL,
        )
    except FileNotFoundError:
        sys.exit(f"\n  ERROR: file '{CSV_FILE}' not found.\n"
                 f"  Set CSV_FILE at the top of this script.")

    n_cols = df.shape[1]
    if n_cols % 2 != 0:
        sys.exit(f"\n  ERROR: expected an even number of columns "
                 f"(VDS,ID pairs), found {n_cols}.")

    # V_GS values sit at column positions 1, 3, 5, … in the header row.
    # pandas stores them as column names after reading.
    try:
        vgs_values = [float(df.columns[i]) for i in range(1, n_cols, 2)]
    except ValueError:
        sys.exit("\n  ERROR: could not parse V_GS values from the CSV header.\n"
                 "  Expected numeric tokens at positions 1, 3, 5, … of row 0.")

    # Rename all columns to unambiguous labels
    new_names = []
    for vgs in vgs_values:
        new_names += [f"VDS_{vgs}", f"ID_{vgs}"]
    df.columns = new_names

    scale = 1e-3 if ID_UNIT.strip().lower() == "ma" else 1.0

    curves = {}
    for vgs in vgs_values:
        sub = df[[f"VDS_{vgs}", f"ID_{vgs}"]].dropna().astype(float)
        vds = sub[f"VDS_{vgs}"].values
        iD  = sub[f"ID_{vgs}"].values * scale     # always stored in Amperes
        order = np.argsort(vds)
        curves[vgs] = (vds[order], iD[order])

    print(f"  V_GS values detected: {[f'{v:.1f}V' for v in vgs_values]}")
    print("  Data summary:")
    for vgs in vgs_values:
        vds, iD = curves[vgs]
        print(f"    V_GS={vgs:.1f} V :  {len(vds):3d} pts  |  "
              f"V_DS ∈ [{vds.min():.2f}, {vds.max():.2f}] V  |  "
              f"I_D ∈ [{iD.min():.4f}, {iD.max():.4f}] A")
    return curves, vgs_values


# =============================================================================
# 2.  SATURATION REGION DETECTION
# =============================================================================

def saturation_indices(vds, iD, frac=SAT_FRAC, min_pts=MIN_SAT_PTS):
    """
    Return indices where I_D >= frac * max(I_D).
    Threshold is relaxed progressively when too few points are found.
    """
    I_max = np.max(iD)
    if I_max <= 0:
        return np.array([], dtype=int)
    for thr in [frac, max(frac - 0.10, 0.60), 0.50]:
        idx = np.where(iD >= thr * I_max)[0]
        if len(idx) >= min_pts:
            return idx
    return np.array([], dtype=int)


# =============================================================================
# 3.  SATURATION FITS:  I_D = I0 * (1 + lambda * V_DS)
# =============================================================================

def fit_saturation(curves, vgs_values):
    fits = {}
    print("\n─── Saturation-region linear fits ──────────────────────────────────")
    for vgs in vgs_values:
        vds, iD = curves[vgs]
        idx = saturation_indices(vds, iD)
        n = len(idx)
        if n < MIN_SAT_PTS:
            print(f"  V_GS={vgs:.1f} V : ⚠  only {n} point(s) — skipping.")
            continue
        slope, intercept, r, _, _ = linregress(vds[idx], iD[idx])
        I0  = intercept if intercept > 0 else float(np.mean(iD[idx]))
        lam = slope / I0
        fits[vgs] = dict(I0=I0, lam=lam, slope=slope, R2=r**2,
                         vds_sat=vds[idx], iD_sat=iD[idx])
        print(f"  V_GS={vgs:.1f} V :  I₀={I0:.4f} A  |  "
              f"λ={lam:.5f} V⁻¹  |  R²={r**2:.4f}  |  n={n}")
    return fits


# =============================================================================
# 4.  CLASSIFY CURVES  — clean (isothermal) vs thermally distorted
# =============================================================================

def classify_curves(fits):
    """
    A curve is flagged as distorted when its lambda exceeds
    LAMBDA_RATIO_WARN times the minimum lambda across all fitted curves.
    High lambda relative to the others is the signature of self-heating
    during a DC sweep at high power.
    """
    if not fits:
        return [], []
    lambdas = {vgs: fits[vgs]["lam"] for vgs in fits}
    min_lam  = max(min(lambdas.values()), 1e-10)
    clean   = sorted(v for v in fits if lambdas[v] <= LAMBDA_RATIO_WARN * min_lam)
    dirty   = sorted(v for v in fits if lambdas[v] >  LAMBDA_RATIO_WARN * min_lam)
    return clean, dirty


# =============================================================================
# 5.  PARAMETER EXTRACTION
#     Primary:  multi-point √(I₀) vs V_GS linear regression
#     Fallback: saturation-knee method (single clean curve)
# =============================================================================

def find_knee_vds(vds, iD, frac=KNEE_FRAC):
    """First V_DS where I_D reaches frac * max(I_D)  →  saturation onset."""
    I_max = np.max(iD)
    if I_max <= 0:
        return None
    for v, i in zip(vds, iD):
        if i >= frac * I_max:
            return float(v)
    return None


def extract_main_params(fits, curves, vgs_values):
    """
    Multi-point regression:  √(I₀) = √K * (V_GS − VTO)
    Written as a linear function of V_GS:
        √(I₀) = m * V_GS + b   with  m = √K,  b = −√K * VTO
        → VTO = −b / m,   K = m²,   KP_eff = 2K

    Only "clean" (isothermal) curves are used for the fit.
    If VTO falls outside VTO_MIN…VTO_MAX the knee fallback is activated.
    """
    warnings    = []
    vto_method  = "regression"
    R2_reg      = None

    clean_vgs, dirty_vgs = classify_curves(fits)

    if dirty_vgs:
        warnings.append(
            f"  ⚠  Thermally distorted curves (λ >> median): "
            f"V_GS = {[f'{v:.1f}V' for v in dirty_vgs]}"
        )
        warnings.append(
            "  ⚠  These are excluded from VTO / K fitting — "
            "repeat those sweeps with short pulses (≤ 20 µs) to fix."
        )

    # Gather clean curves that have a valid (positive) I₀
    valid_vgs = [v for v in clean_vgs if fits[v]["I0"] > 0]

    if len(valid_vgs) < 2:
        warnings.append("  ⚠  Fewer than 2 clean curves available — falling back to knee method.")
        vto_method = "knee"
    else:
        vgs_arr  = np.array(valid_vgs, dtype=float)
        sqrt_I0  = np.array([np.sqrt(fits[v]["I0"]) for v in valid_vgs])
        m, b, r, _, _ = linregress(vgs_arr, sqrt_I0)
        R2_reg   = r ** 2
        VTO_reg  = (-b / m) if m > 0 else -999.0

        if VTO_MIN < VTO_reg < VTO_MAX:
            VTO     = VTO_reg
            K       = m ** 2
            KP_eff  = 2.0 * K
            LAMBDA  = float(np.mean([fits[v]["lam"] for v in clean_vgs]))
        else:
            warnings.append(
                f"  ⚠  Regression VTO = {VTO_reg:.3f} V is outside "
                f"[{VTO_MIN}, {VTO_MAX}] V — activating knee fallback."
            )
            vto_method = "knee"

    if vto_method == "knee":
        # Use the lowest-power clean curve — least affected by self-heating
        ref_vgs  = min(clean_vgs)
        vds_k, iD_k = curves[ref_vgs]
        vds_knee = find_knee_vds(vds_k, iD_k)
        VTO      = ref_vgs - vds_knee if vds_knee is not None else 3.0
        K        = fits[ref_vgs]["I0"] / (ref_vgs - VTO) ** 2
        KP_eff   = 2.0 * K
        LAMBDA   = fits[ref_vgs]["lam"]
        warnings.append(
            f"  ⚠  Knee fallback: used V_GS={ref_vgs:.1f}V curve only."
        )

    return VTO, KP_eff, K, LAMBDA, vto_method, clean_vgs, dirty_vgs, warnings, R2_reg


# =============================================================================
# 6.  RDS(on) FROM DEEP-LINEAR REGION
# =============================================================================

def extract_rds_on(curves, vgs_values, frac=LIN_FRAC):
    """Use the highest available V_GS: deepest turn-on → lowest resistance."""
    vgs_max = max(v for v in vgs_values)
    vds, iD = curves[vgs_max]
    thresh  = frac * vds.max()
    idx     = np.where((vds > 0.0) & (vds <= thresh))[0]
    if len(idx) < 2:
        return None, vgs_max
    slope, _, _, _, _ = linregress(vds[idx], iD[idx])
    return ((1.0 / slope) if slope > 0 else None), vgs_max


# =============================================================================
# 7.  LEVEL-1 MODEL CURVE  (for verification plots)
# =============================================================================

def level1_curve(vds, vgs, VTO, KP_eff, LAMBDA):
    iD = np.zeros_like(vds, dtype=float)
    if vgs <= VTO:
        return iD
    vdsat = vgs - VTO
    lin   = (vds >= 0.0) & (vds < vdsat)
    sat   = vds >= vdsat
    iD[lin] = KP_eff * ((vgs-VTO)*vds[lin] - 0.5*vds[lin]**2) * (1+LAMBDA*vds[lin])
    iD[sat] = 0.5 * KP_eff * vdsat**2 * (1+LAMBDA*vds[sat])
    return np.maximum(iD, 0.0)


# =============================================================================
# 8.  PRINT SUMMARY AND MODEL CARD
# =============================================================================

def print_results(VTO, KP_eff, K, LAMBDA, fits, RDS, rds_vgs,
                  vto_method, clean_vgs, dirty_vgs, warnings, R2_reg):
    bar = "═" * 68

    if warnings:
        print(f"\n{bar}")
        print("  ⚠  DATA QUALITY WARNINGS")
        print(bar)
        for w in warnings:
            print(w)

    print(f"\n{bar}")
    print("  EXTRACTED PARAMETERS")
    print(bar)

    if vto_method == "regression":
        method_note = (f"multi-point √(I₀) vs V_GS regression  "
                       f"(R² = {R2_reg:.4f},  n = {len(clean_vgs)} curves)")
    else:
        method_note = "saturation-knee fallback (see warnings)"

    print(f"  VTO        =  {VTO:.4f}      V   [{method_note}]")
    print(f"  KP_eff (β) =  {KP_eff:.6f} A/V²  (= 2K, for W=L=1 on the instance)")
    print(f"  K  (= β/2) =  {K:.6f} A/V²")
    print(f"  LAMBDA     =  {LAMBDA:.6f} V⁻¹  "
          f"(avg over V_GS = {[f'{v:.1f}V' for v in clean_vgs]})")

    print(f"\n  Saturation intercepts I₀ per curve:")
    for vgs in sorted(fits.keys()):
        tag = "  ✓ clean" if vgs in clean_vgs else "  ✗ distorted — excluded"
        print(f"    V_GS={vgs:.1f} V :  I₀ = {fits[vgs]['I0']:.4f} A  "
              f"|  λ = {fits[vgs]['lam']:.5f} V⁻¹  "
              f"|  R² = {fits[vgs]['R2']:.4f}{tag}")

    if RDS is not None:
        print(f"\n  RDS(on) ≈ {RDS*1e3:.2f} mΩ  "
              f"(V_GS={rds_vgs:.1f}V, deep-linear region)")

    print(bar)
    print(f"\n{bar}")
    print("  SPICE LEVEL 1 MODEL CARD")
    print(bar)
    print(f"  * Device  : {DEVICE_NAME}  (N-channel enhancement MOSFET)")
    print(f"  * Curves used for extraction: "
          f"V_GS = {[f'{v:.1f}V' for v in clean_vgs]}")
    if dirty_vgs:
        print(f"  * Excluded (self-heating)   : "
              f"V_GS = {[f'{v:.1f}V' for v in dirty_vgs]}")
    print(f"  *")
    print(f"  * KP is EFFECTIVE = KP_intrinsic × (W/L).  "
          f"Always use W=1 L=1 on each instance.")
    print(f"  * Capacitances (CGSO, CGDO, CJ …) are NOT extracted here.")
    print()
    print(f"  .model {DEVICE_NAME} NMOS (Level=1")
    print(f"  +  VTO    = {VTO:.4f}      $ threshold voltage  [V]")
    print(f"  +  KP     = {KP_eff:.6f}  $ eff. transconductance [A/V²]  (W=L=1)")
    print(f"  +  LAMBDA = {LAMBDA:.6f}  $ channel-length modulation  [V⁻¹]")
    if RDS is not None:
        print(f"  +  RD     = {RDS/2:.5f}    $ ≈ RDS(on)/2  [Ω]")
        print(f"  +  RS     = {RDS/2:.5f}    $ ≈ RDS(on)/2  [Ω]")
    print(f"  +  )")
    print()
    print(f"  * M1  Drain  Gate  Source  Bulk  {DEVICE_NAME}  W=1  L=1")
    print(bar)


# =============================================================================
# 9.  PLOTS
# =============================================================================

def make_plots(curves, vgs_values, fits, VTO, KP_eff, LAMBDA,
               vto_method, clean_vgs, dirty_vgs, R2_reg):

    n     = len(vgs_values)
    cmap  = plt.cm.plasma
    colors = {vgs: cmap(i / max(n - 1, 1)) for i, vgs in enumerate(vgs_values)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle(f"{DEVICE_NAME}  –  SPICE Level 1 Parameter Extraction",
                 fontsize=13, fontweight="bold")

    # ── Left: output characteristics ──────────────────────────────────────────
    vds_max  = max(curves[v][0].max() for v in vgs_values)
    vds_fine = np.linspace(0.0, vds_max * 1.02, 600)

    for vgs in vgs_values:
        vds, iD = curves[vgs]
        c   = colors[vgs]
        ls  = "-" if vgs in clean_vgs else ":"    # dotted model for distorted

        ax1.plot(vds, iD, "o", color=c, ms=3, zorder=3,
                 label=f"Data  V_GS={vgs:.1f} V")

        # Mark saturation region with open squares
        if vgs in fits:
            ii = saturation_indices(vds, iD)
            if len(ii):
                ax1.plot(vds[ii], iD[ii], "s", color=c,
                         ms=6, mfc="none", mew=1.4, zorder=4)

        # Level-1 model overlay
        iD_m = level1_curve(vds_fine, vgs, VTO, KP_eff, LAMBDA)
        ax1.plot(vds_fine, iD_m, ls, color=c, lw=1.7, zorder=2,
                 label=f"Model V_GS={vgs:.1f} V")

    ax1.set_xlabel("V$_{DS}$  [V]",  fontsize=11)
    ax1.set_ylabel("I$_D$  [A]",     fontsize=11)
    ax1.set_title("Output Characteristics\n"
                  "data ●   sat. region ◻   model (— clean  ·· excluded)",
                  fontsize=9.5)
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.set_xlim(left=0); ax1.set_ylim(bottom=0)

    # ── Right: √(I₀) vs V_GS  — multi-point VTO extraction ──────────────────
    all_vgs_fit = sorted(fits.keys())

    # Clean points
    c_vgs = [v for v in all_vgs_fit if v in clean_vgs]
    c_sq  = [np.sqrt(fits[v]["I0"]) for v in c_vgs]
    ax2.plot(c_vgs, c_sq, "D", color="#1565C0", ms=9, zorder=5,
             label="√(I₀) — clean curves (used)")

    # Distorted points
    if dirty_vgs:
        d_vgs = [v for v in all_vgs_fit if v in dirty_vgs]
        d_sq  = [np.sqrt(fits[v]["I0"]) for v in d_vgs]
        ax2.plot(d_vgs, d_sq, "x", color="#C62828", ms=11,
                 mew=2.5, zorder=5, label="√(I₀) — distorted (excluded)")

    # Regression line extended down to VTO
    if len(c_vgs) >= 2:
        m, b, _, _, _ = linregress(np.array(c_vgs, dtype=float),
                                   np.array(c_sq))
        VTO_graph = -b / m
        x_line    = np.linspace(min(VTO, VTO_graph) * 0.80,
                                max(all_vgs_fit) * 1.05, 300)
        r2_txt = f"  R² = {R2_reg:.4f}" if R2_reg is not None else ""
        ax2.plot(x_line, m * x_line + b, "--k", lw=1.6,
                 label=f"Regression → VTO = {VTO:.3f} V{r2_txt}")
        ax2.axvline(VTO, color="#C62828", lw=1.5, ls=":",
                    label=f"VTO = {VTO:.3f} V")

    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_xlabel("V$_{GS}$  [V]",       fontsize=11)
    ax2.set_ylabel("$\\sqrt{I_0}$  [√A]", fontsize=11)
    ax2.set_title("Threshold Voltage Extraction\n"
                  "√(I₀) vs V$_{GS}$  –  multi-point regression  →  VTO",
                  fontsize=9.5)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)
    if all_vgs_fit:
        ax2.set_xlim(left=min(all_vgs_fit) * 0.80)
    ax2.set_ylim(bottom=min(0.0,
                 min(m * x_line[0] + b, 0) * 1.15) if len(c_vgs) >= 2 else 0)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved  →  {PLOT_FILE}")
    plt.show()


# =============================================================================
#  MAIN
# =============================================================================

def main():
    bar = "=" * 68
    print(bar)
    print(f"  MOSFET SPICE Level 1 Parameter Extractor  –  {DEVICE_NAME}")
    print(bar)

    curves, vgs_values = load_data()
    fits = fit_saturation(curves, vgs_values)

    if len(fits) < 2:
        sys.exit("\n  ERROR: Need at least 2 curves with valid saturation fits.")

    (VTO, KP_eff, K, LAMBDA, vto_method,
     clean_vgs, dirty_vgs, warnings, R2_reg) = extract_main_params(
         fits, curves, vgs_values)

    RDS, rds_vgs = extract_rds_on(curves, vgs_values)

    print_results(VTO, KP_eff, K, LAMBDA, fits, RDS, rds_vgs,
                  vto_method, clean_vgs, dirty_vgs, warnings, R2_reg)

    make_plots(curves, vgs_values, fits, VTO, KP_eff, LAMBDA,
               vto_method, clean_vgs, dirty_vgs, R2_reg)


if __name__ == "__main__":
    main()
