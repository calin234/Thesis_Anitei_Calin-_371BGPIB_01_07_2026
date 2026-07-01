import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

#!/usr/bin/env python3
"""
=============================================================================
  MOSFET SPICE Level 1 Parameter Extractor
  -----------------------------------------
  Extracts VTO, KP (effective), LAMBDA, and RDS(on) from MOSFET output
  characteristics  I_D vs V_DS  at multiple fixed V_GS values.

  Input CSV structure
  -------------------
  Row 0  : header  e.g. "Gate Voltage, 0, Gate Voltage, 5, Gate Voltage, 10"
  Rows 1+: 6 data columns in this order:
             VDS[V] | ID[mA] | VDS[V] | ID[mA] | VDS[V] | ID[mA]
             ←  V_GS=0V  →  ←  V_GS=5V  →  ← V_GS=10V  →

  SPICE Level 1 (Shichman-Hodges) equations used
  -----------------------------------------------
  Saturation  (V_DS >= V_GS - VTO):
    I_D = (KP/2) × (W/L) × (V_GS - VTO)² × (1 + LAMBDA × V_DS)

  Linear  (0 <= V_DS < V_GS - VTO):
    I_D = KP × (W/L) × [(V_GS-VTO)×V_DS − V_DS²/2] × (1 + LAMBDA×V_DS)

  Note on KP
  ----------
  W and L are unknown for this device, so only the product
      KP_eff  =  KP_intrinsic × (W/L)
  is extracted.  In the .model card KP = KP_eff; use  W=1  L=1  on every
  MOSFET instance in the netlist so SPICE computes:
      I_D_sat = (KP_eff/2) × (V_GS − VTO)²  ×  (1 + LAMBDA × V_DS)

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

CSV_FILE    = "date04_06bun.csv"   # path to your CSV data file
DEVICE_NAME = "IRF740"                   # used in labels and model card

VGS_VALUES  = [0, 5, 10]                # gate voltages present in the file [V]
ID_UNIT     = "mA"                      # current unit in the CSV: "mA" or "A"

# CSV format (adjust if the file uses semicolons or comma-decimals)
CSV_SEP     = None    # None = auto-detect (handles "," and ";")
CSV_DECIMAL = "."     # decimal separator: "." for US/standard, "," for Romanian/EU

# Saturation region detection
SAT_FRAC    = 0.85    # points with I_D >= SAT_FRAC × max(I_D) → in saturation
MIN_SAT_PTS = 3       # minimum saturation points required for a valid fit

# Deep-linear region bound used for RDS(on) extraction (fraction of max V_DS)
LIN_FRAC    = 0.12

# Output plot filename (PNG)
PLOT_FILE   = f"{DEVICE_NAME}_SPICE_extraction.png"

# Per-curve colours
COLORS = {0: "#78909C", 5: "#1565C0", 10: "#C62828"}

# =============================================================================
# 1.  LOAD DATA
# =============================================================================

def load_data():
    """Read CSV, assign column names, convert I_D to Amperes."""
    print(f"\n  Reading '{CSV_FILE}' …")
    try:
        df = pd.read_csv(
            CSV_FILE,
            header=0,
            sep=CSV_SEP,
            engine="python" if CSV_SEP is None else "c",
            decimal=CSV_DECIMAL,
        )
    except FileNotFoundError:
        sys.exit(f"\n  ERROR: file '{CSV_FILE}' not found.\n"
                 f"  Set CSV_FILE at the top of this script.")

    if df.shape[1] != 6:
        sys.exit(f"\n  ERROR: expected 6 columns, found {df.shape[1]}.\n"
                 f"  Check CSV_FILE structure.")

    # Rename columns regardless of original header text
    # NOTE: CSV layout is V_DS first, then I_D, for each V_GS block
    df.columns = ["VDS_0", "ID_0", "VDS_5", "ID_5", "VDS_10", "ID_10"]

    scale = 1e-3 if ID_UNIT.strip().lower() == "ma" else 1.0

    curves = {}
    for vgs in VGS_VALUES:
        v = int(vgs)
        sub = df[[f"ID_{v}", f"VDS_{v}"]].dropna().astype(float)
        vds = sub[f"VDS_{v}"].values
        iD  = sub[f"ID_{v}"].values * scale
        order = np.argsort(vds)
        curves[vgs] = (vds[order], iD[order])

    print("  Data loaded – summary:")
    for vgs in VGS_VALUES:
        vds, iD = curves[vgs]
        print(f"    V_GS={vgs:2d} V :  {len(vds):3d} pts  |  "
              f"V_DS in [{vds.min():.2f}, {vds.max():.2f}] V  |  "
              f"I_D  in [{iD.min()*1e3:.3f}, {iD.max()*1e3:.3f}] mA")
    return curves


# =============================================================================
# 2.  SATURATION REGION DETECTION
# =============================================================================

def saturation_indices(vds, iD, frac=SAT_FRAC, min_pts=MIN_SAT_PTS):
    """
    Return indices of points considered to be in the saturation region.

    Criterion: I_D >= frac × max(I_D).  The threshold is relaxed
    progressively if too few points are found.
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
# 3.  SATURATION-REGION LINEAR FIT:  I_D = I₀ × (1 + λ × V_DS)
# =============================================================================

def fit_saturation(curves):
    """
    For V_GS in {5, 10} V, fit a line through the saturation points:
        I_D = I₀ + slope × V_DS

    From this:
        I₀    ≈ K × (V_GS − VTO)²     (intercept at V_DS = 0)
        LAMBDA = slope / I₀             (channel-length modulation)
    """
    fits = {}
    print("\n─── Saturation-region linear fits ──────────────────────────────────")
    for vgs in [5, 10]:
        vds, iD = curves[vgs]
        idx = saturation_indices(vds, iD)
        n = len(idx)

        if n < MIN_SAT_PTS:
            print(f"  V_GS={vgs:2d} V : ⚠  only {n} point(s) found – skipping.")
            continue

        slope, intercept, r, _, _ = linregress(vds[idx], iD[idx])

        # Safety: extrapolated intercept must be positive
        I0 = intercept if intercept > 0 else float(np.mean(iD[idx]))
        lam = slope / I0

        fits[vgs] = dict(I0=I0, lam=lam, slope=slope, R2=r**2,
                         vds_sat=vds[idx], iD_sat=iD[idx])
        print(f"  V_GS={vgs:2d} V :  I₀ = {I0*1e3:.4f} mA  |  "
              f"λ = {lam:.5f} V⁻¹  |  R² = {r**2:.4f}  |  n = {n}")
    return fits


# =============================================================================
# 4.  EXTRACT  VTO, KP_eff, LAMBDA  –  with self-heating fallback
# =============================================================================

# Fraction of max I_D used to locate the saturation knee (for VTO fallback)
KNEE_FRAC = 0.85

# VTO validity range: must be between these bounds to be considered physical
VTO_MIN, VTO_MAX = 0.5, 5.0

# Lambda consistency check: if λ(10V)/λ(5V) exceeds this ratio, warn self-heating
LAMBDA_RATIO_WARN = 5.0


def find_knee_vds(vds, iD, frac=KNEE_FRAC):
    """
    Return the V_DS value at which I_D first reaches frac × max(I_D).
    This marks the approximate onset of saturation, from which
    VTO ≈ V_GS − V_DS_knee.
    """
    I_max = np.max(iD)
    if I_max <= 0:
        return None
    for v, i in zip(vds, iD):
        if i >= frac * I_max:
            return float(v)
    return None


def extract_main_params(fits, curves):
    """
    Primary path (isothermal): use the current-ratio formula
        r = √(I₀_5 / I₀_10) = (5 − VTO) / (10 − VTO)
        → VTO = (5 − 10·r) / (1 − r)

    This requires both curves to be measured at the same temperature.
    When it fails (VTO non-physical, e.g. negative), the most common cause
    is self-heating of the high-power curve (V_GS = 10 V) during DC sweep.

    Fallback path (self-heating detected): estimate VTO from the saturation
    knee of each curve (V_DS_knee ≈ V_GS − VTO), then derive K and LAMBDA
    exclusively from the lower-power V_GS = 5 V curve, which is far less
    affected by thermal drift.
    """
    I0_5  = fits[5]["I0"]
    I0_10 = fits[10]["I0"]

    # ── Attempt primary extraction ────────────────────────────────────────────
    r       = np.sqrt(I0_5 / I0_10)
    VTO_rat = (5.0 - 10.0 * r) / (1.0 - r)

    lam5  = fits[5]["lam"]
    lam10 = fits[10]["lam"]
    lam_ratio = lam10 / lam5 if lam5 > 0 else float("inf")

    self_heating_warning = []

    # Detect self-heating: VTO non-physical OR lambda values wildly inconsistent
    vto_bad    = not (VTO_MIN < VTO_rat < VTO_MAX)
    lambda_bad = lam_ratio > LAMBDA_RATIO_WARN

    if vto_bad or lambda_bad:
        # ── Fallback: knee-based VTO ──────────────────────────────────────────
        if vto_bad:
            self_heating_warning.append(
                f"  ⚠  Ratio-based VTO = {VTO_rat:.2f} V is non-physical (expected {VTO_MIN}–{VTO_MAX} V)."
            )
        if lambda_bad:
            self_heating_warning.append(
                f"  ⚠  λ(10V)/λ(5V) = {lam_ratio:.1f}×  (>{LAMBDA_RATIO_WARN}×) — likely DC self-heating."
            )
        self_heating_warning += [
            "  ⚠  The V_GS = 10 V curve appears thermally distorted (DC sweep at high power).",
            "  ⚠  Falling back to knee-detection VTO and V_GS = 5 V saturation for K and λ.",
            "  ⚠  For accurate extraction, repeat the V_GS = 10 V sweep with short pulses (≤ 20 µs).",
        ]

        # Estimate VTO from the saturation knee of each curve.
        # Print both for diagnostics, but only the V_GS=5V knee is used:
        # at high power, the V_GS=10V "knee" is itself shifted by
        # self-heating and is not a reliable estimator of V_DSAT = V_GS-VTO.
        knee_estimates = {}
        print("\n─── Knee-based VTO estimation (self-heating fallback) ───────────────")
        for vgs in [5, 10]:
            vds, iD = curves[vgs]
            vds_knee = find_knee_vds(vds, iD)
            if vds_knee is not None:
                vto_k = vgs - vds_knee
                knee_estimates[vgs] = vto_k
                tag = "  (used)" if vgs == 5 else "  (reference only — distorted by heating)"
                print(f"  V_GS={vgs:2d} V : V_DS_knee ≈ {vds_knee:.2f} V  →  VTO ≈ {vto_k:.2f} V{tag}")

        if 5 in knee_estimates:
            VTO = knee_estimates[5]
        elif knee_estimates:
            VTO = float(np.mean(list(knee_estimates.values())))
        else:
            VTO = 3.0   # last-resort default (typical for IRF-series power MOSFETs)

        # K and LAMBDA from V_GS = 5 V only (the clean, low-power curve)
        K5  = I0_5 / (5.0 - VTO) ** 2
        K10 = K5                    # set equal; flag in output that only 5V used
        K   = K5

        LAMBDA = lam5               # use only V_GS = 5 V lambda
        vto_method = "knee"

    else:
        # ── Primary path succeeded ────────────────────────────────────────────
        VTO = VTO_rat
        K5  = I0_5  / (5.0  - VTO) ** 2
        K10 = I0_10 / (10.0 - VTO) ** 2
        K   = 0.5 * (K5 + K10)
        LAMBDA = 0.5 * (lam5 + lam10)
        vto_method = "ratio"

    KP_eff = 2.0 * K

    return VTO, KP_eff, K5, K10, LAMBDA, vto_method, self_heating_warning


# =============================================================================
# 5.  RDS(on) FROM DEEP-LINEAR REGION AT V_GS = 10 V
# =============================================================================

def extract_rds_on(curves, vgs=10, frac=LIN_FRAC):
    """
    Estimate R_DS(on) as 1 / (∂I_D/∂V_DS) in the deep-linear region
    (small V_DS, V_GS = 10 V).
    """
    vds, iD = curves[vgs]
    thresh  = frac * vds.max()
    idx     = np.where((vds > 0.0) & (vds <= thresh))[0]
    if len(idx) < 2:
        return None
    slope, _, _, _, _ = linregress(vds[idx], iD[idx])
    return (1.0 / slope) if slope > 0 else None


# =============================================================================
# 6.  LEVEL-1 MODEL CURVE  (for verification plots)
# =============================================================================

def level1_curve(vds, vgs, VTO, KP_eff, LAMBDA):
    """Compute I_D [A] using the SPICE Level-1 equations."""
    iD = np.zeros_like(vds, dtype=float)
    if vgs <= VTO:
        return iD
    vdsat = vgs - VTO
    lin = (vds >= 0.0) & (vds < vdsat)
    sat = vds >= vdsat
    v_l = vds[lin];  v_s = vds[sat]
    iD[lin] = KP_eff * ((vgs - VTO) * v_l - 0.5 * v_l**2) * (1.0 + LAMBDA * v_l)
    iD[sat] = 0.5 * KP_eff * vdsat**2 * (1.0 + LAMBDA * v_s)
    return np.maximum(iD, 0.0)


# =============================================================================
# 7.  PRINT SUMMARY AND SPICE MODEL CARD
# =============================================================================

def print_results(VTO, KP_eff, K5, K10, LAMBDA, fits, RDS, vto_method, warnings):
    lam5, lam10 = fits[5]["lam"], fits[10]["lam"]
    K = 0.5 * KP_eff

    bar = "═" * 66

    if warnings:
        print(f"\n{bar}")
        print("  ⚠  DATA QUALITY WARNING")
        print(bar)
        for w in warnings:
            print(w)

    print(f"\n{bar}")
    print("  EXTRACTED PARAMETERS")
    print(bar)
    method_note = "ratio formula (isothermal)" if vto_method == "ratio" else "saturation-knee fallback"
    print(f"  VTO         =  {VTO:.4f}       V   [{method_note}]")
    print(f"  K  (= β/2)  =  {K*1e3:.5f}  mA/V²  (average)")
    print(f"                  {K5*1e3:.5f}  mA/V²  from V_GS = 5 V")
    print(f"                  {K10*1e3:.5f}  mA/V²  from V_GS = 10 V"
          + ("  (set equal to 5V value, see warning)" if vto_method == "knee" else ""))
    print(f"  KP_eff (β)  =  {KP_eff:.6f}  A/V²   (= 2K, for W=L=1)")
    print(f"  LAMBDA      =  {LAMBDA:.6f}  V⁻¹"
          + ("   (from V_GS = 5 V only, see warning)" if vto_method == "knee" else "   (average)"))
    print(f"                  {lam5:.6f}  V⁻¹   from V_GS = 5 V")
    print(f"                  {lam10:.6f}  V⁻¹   from V_GS = 10 V"
          + ("  (excluded — thermally distorted)" if vto_method == "knee" else ""))
    if RDS:
        print(f"  RDS(on)     ≈  {RDS*1e3:.2f} mΩ  (V_GS=10V, deep-linear region)")
    print(bar)

    print(f"\n{bar}")
    print("  SPICE LEVEL 1 MODEL CARD")
    print(bar)
    print(f"  * Device : {DEVICE_NAME}  (N-channel enhancement MOSFET)")
    print(f"  *")
    print(f"  * KP here is EFFECTIVE = KP_intrinsic × (W/L).")
    print(f"  * ⚠  Always use  W=1  L=1  on every MOSFET instance in the netlist.")
    print(f"  *    Capacitances (CGSO, CGDO, CJ, …) are NOT extracted here;")
    print(f"  *    add them from the datasheet for transient/AC simulations.")
    print()
    print(f"  .model {DEVICE_NAME} NMOS (Level=1")
    print(f"  +  VTO    = {VTO:.4f}      $ threshold voltage  [V]")
    print(f"  +  KP     = {KP_eff:.6f}  $ eff. transconductance KP*(W/L)  [A/V²]")
    print(f"  +  LAMBDA = {LAMBDA:.6f}  $ channel-length modulation  [V⁻¹]")
    if RDS:
        rds_half = RDS / 2.0
        print(f"  +  RD     = {rds_half:.5f}   $ approx. drain resistance ≈ RDS(on)/2  [Ω]")
        print(f"  +  RS     = {rds_half:.5f}   $ approx. source resistance ≈ RDS(on)/2 [Ω]")
    print(f"  +  )")
    print()
    print(f"  * Instance syntax (PSpice / LTspice):")
    print(f"  * M1  Drain  Gate  Source  Bulk  {DEVICE_NAME}  W=1  L=1")
    print(bar)


# =============================================================================
# 8.  DIAGNOSTIC PLOTS
# =============================================================================

def make_plots(curves, fits, VTO, KP_eff, LAMBDA, vto_method):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"{DEVICE_NAME}  –  SPICE Level 1 Parameter Extraction",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Plot 1: Output characteristics  ─────────────────────────────────────
    vds_max  = max(curves[5][0].max(), curves[10][0].max())
    vds_fine = np.linspace(0.0, vds_max * 1.05, 500)

    for vgs in VGS_VALUES:
        vds, iD = curves[vgs]
        c = COLORS[vgs]
        label_data = f"Data  V_GS = {vgs} V"

        # Raw data points
        ax1.plot(vds, iD * 1e3, "o", color=c, ms=4, zorder=3,
                 label=label_data)

        # Highlight saturation region with open squares
        if vgs in fits:
            ii = saturation_indices(vds, iD)
            if len(ii):
                ax1.plot(vds[ii], iD[ii] * 1e3, "s", color=c,
                         ms=7, mfc="none", mew=1.5, zorder=4)

        # Level-1 model curve (dashed)
        if vgs > 0:
            iD_model = level1_curve(vds_fine, vgs, VTO, KP_eff, LAMBDA)
            ax1.plot(vds_fine, iD_model * 1e3, "--", color=c, lw=1.8,
                     zorder=2, label=f"Model V_GS = {vgs} V")

    ax1.set_xlabel("V$_{DS}$  [V]",  fontsize=11)
    ax1.set_ylabel("I$_{D}$  [mA]", fontsize=11)
    ax1.set_title(
        "Output Characteristics\n"
        "data  ●    saturation region  ◻    Level-1 model  ╌╌",
        fontsize=9.5,
    )
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.set_xlim(left=0.0)
    ax1.set_ylim(bottom=0.0)

    # ── Plot 2: VTO extraction quality ────────────────────────────────────
    if vto_method == "ratio":
        # √(I₀) vs V_GS  →  linear extrapolation to x-axis gives VTO
        vgs_arr = np.array([5.0, 10.0])
        I0_arr  = np.array([fits[v]["I0"] for v in [5, 10]])
        sqI0    = np.sqrt(I0_arr * 1e3)          # √mA  (plot in √mA for readability)

        m_gr, b_gr, _, _, _ = linregress(vgs_arr, sqI0)
        VTO_graph = -b_gr / m_gr

        vgs_line = np.linspace(min(VTO, VTO_graph) * 0.85, 11.5, 200)

        ax2.plot(vgs_arr, sqI0, "D", color="#1565C0", ms=9, zorder=5,
                 label="√(I₀) from saturation fit")
        ax2.plot(vgs_line, m_gr * vgs_line + b_gr, "--k", lw=1.6,
                 label=f"Linear extrapolation  →  VTO = {VTO_graph:.3f} V")
        ax2.axvline(VTO, color="#C62828", lw=1.8, ls=":",
                    label=f"VTO (analytical) = {VTO:.3f} V")
        ax2.axhline(0.0, color="black", lw=0.8, ls="-")

        ax2.set_xlabel("V$_{GS}$  [V]",         fontsize=11)
        ax2.set_ylabel("$\\sqrt{I_{D,sat}}$  [√mA]", fontsize=11)
        ax2.set_title(
            "Threshold Voltage Extraction\n"
            "√(I₀) vs V$_{GS}$  –  linear extrapolation  →  VTO",
            fontsize=9.5,
        )
        ax2.legend(fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.5)
        y_min = min(0.0, m_gr * vgs_line[0] + b_gr) * 1.15
        ax2.set_ylim(bottom=y_min)

    else:
        # Knee-based fallback: show knee points V_DS_knee = V_GS - VTO.
        # Only the V_GS = 5V knee is used for VTO; the V_GS = 10V knee is
        # shown only as a (distorted) reference.
        for vgs, marker, label in [(5, "D", "V_GS=5V knee  (used for VTO)"),
                                     (10, "x", "V_GS=10V knee  (reference, heat-distorted)")]:
            vds, iD = curves[vgs]
            vk = find_knee_vds(vds, iD)
            if vk is not None:
                color = "#1565C0" if vgs == 5 else "#C62828"
                ax2.plot([vgs], [vk], marker, color=color, ms=10, mew=2, zorder=5,
                         label=label)

        vgs_line = np.linspace(0.0, 11.5, 200)
        # Reference line: V_DS_knee = V_GS - VTO  (slope 1, intercept -VTO)
        ax2.plot(vgs_line, vgs_line - VTO, "--k", lw=1.6,
                 label=f"V$_{{DS,knee}}$ = V$_{{GS}}$ − VTO   (VTO = {VTO:.3f} V)")
        ax2.axhline(0.0, color="black", lw=0.8, ls="-")

        ax2.set_xlabel("V$_{GS}$  [V]", fontsize=11)
        ax2.set_ylabel("V$_{DS,knee}$  [V]", fontsize=11)
        ax2.set_title(
            "Threshold Voltage Extraction (fallback)\n"
            "Saturation-knee method  –  V$_{DS,knee}$ = V$_{GS}$ − VTO",
            fontsize=9.5,
        )
        ax2.legend(fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.set_xlim(left=0.0)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved  →  {PLOT_FILE}")
    plt.show()


# =============================================================================
#  MAIN
# =============================================================================

def main():
    bar = "=" * 66
    print(bar)
    print(f"  MOSFET SPICE Level 1 Parameter Extractor  –  {DEVICE_NAME}")
    print(bar)

    curves = load_data()
    fits   = fit_saturation(curves)

    missing = [v for v in [5, 10] if v not in fits]
    if missing:
        sys.exit(f"\n  ERROR: Could not fit saturation region for V_GS = {missing} V.\n"
                 f"  Check the data or reduce MIN_SAT_PTS / SAT_FRAC.")

    VTO, KP_eff, K5, K10, LAMBDA, vto_method, warnings = extract_main_params(fits, curves)
    RDS = extract_rds_on(curves)

    print_results(VTO, KP_eff, K5, K10, LAMBDA, fits, RDS, vto_method, warnings)
    make_plots(curves, fits, VTO, KP_eff, LAMBDA, vto_method)


if __name__ == "__main__":
    main()
