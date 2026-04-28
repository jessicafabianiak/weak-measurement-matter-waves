"""
=============================================================================
Python code used during the photon-curtain manuscript review session
=============================================================================

This file collects all the Python computations and figure generation used
to identify mathematical errors in v2 and to determine the corrected
operating point for v3 of the manuscript.

Sections:
  1. Initial check of v2's numerical estimates (showed g, P_sc were wrong)
  2. Verification of the Englert-bound algebra (V = exp(-L/2) violates V^2+D^2 <= 1)
  3. AAV mapping / SNR / tau analysis
  4. Photon-kick derivative and narrow-curtain limit
  5. Operating-point search for the corrected manuscript
  6. Figure 3 (operating regime) regeneration with corrected parameters

Constants used throughout:
  hbar = 1.054571817e-34 J·s
  c    = 2.998e8 m/s
  Rb-87 D2: Gamma/2pi = 6.07 MHz, omega0/2pi = 384 THz, lambda = 780 nm,
            I_sat = 16.7 W/m^2, M = 1.443e-25 kg
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Universal constants
hbar = 1.054571817e-34   # J·s
c    = 2.998e8            # m/s
kB   = 1.381e-23          # J/K

# Rb-87 D2 line
Gamma  = 2*np.pi*6.07e6
omega0 = 2*np.pi*384e12
lam_L  = 780e-9
omega_L = 2*np.pi*c/lam_L
I_sat  = 16.7              # W/m^2 (cycling)
M_Rb   = 1.443e-25         # kg


# =============================================================================
# 1. INITIAL CHECK OF V2 PARAMETERS (showed g and P_sc were wrong)
# =============================================================================
def check_v2_parameters():
    """
    The original v2 manuscript claimed:
      P = 10 nW, Delta/2pi = 10 GHz, w = 2 um, v_z = 1 m/s
      => g ~ 1e-3, P_sc ~ 1e-9
    Verify these numbers.
    """
    print("="*70)
    print("V2 PARAMETER CHECK")
    print("="*70)

    P = 10e-9
    w = 2e-6
    Delta = 2*np.pi*1e10
    v_z = 1.0

    tau_phot = w/c
    tau_atom = w/v_z
    I0 = 2*P/(np.pi*w**2)
    print(f"Peak intensity I0 = {I0:.3e} W/m^2")
    print(f"I0/I_sat = {I0/I_sat:.2f}  (well above saturation)")

    # Rabi frequency from intensity: I/I_sat = 2 Omega^2 / Gamma^2
    Omega0 = Gamma * np.sqrt(I0/(2*I_sat))
    print(f"Omega0 = 2pi x {Omega0/(2*np.pi):.3e} Hz")

    g_phot = Omega0**2 * tau_phot / (4*Delta)
    g_atom = Omega0**2 * tau_atom / (4*Delta)
    print(f"\ng using tau_phot (w/c = {tau_phot:.2e} s): g = {g_phot:.3e}")
    print(f"g using tau_atom (w/v_z = {tau_atom:.2e} s): g = {g_atom:.3e}")
    print(f"Paper claims g ~ 1e-3 -- WRONG, by 6 orders of magnitude")

    # Spontaneous emission probability
    Gamma_sc = Gamma * Omega0**2 / (4*Delta**2)
    P_sc = Gamma_sc * tau_atom
    print(f"\nP_sc per atom transit = {P_sc:.3e}")
    print(f"Paper claims P_sc ~ 1e-9 -- WRONG, off by ~5 orders of magnitude")
    print()


# =============================================================================
# 2. ENGLERT BOUND CHECK
# =============================================================================
def check_englert_bound():
    """
    The original v2 had:
      V = exp(-Lambda_1 / 2)
      D = sqrt(Lambda_1)
    and claimed V^2 + D^2 <= 1.  Check this.
    """
    print("="*70)
    print("ENGLERT BOUND CHECK (V^2 + D^2 <= 1)")
    print("="*70)
    print(f"{'Lambda_1':>10} {'V':>8} {'D':>8} {'V^2+D^2':>10}  Result")
    for L in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        V = np.exp(-L/2)
        D = np.sqrt(L)
        s = V**2 + D**2
        flag = "OK" if s <= 1.0 + 1e-12 else "VIOLATES"
        print(f"{L:>10.2f} {V:>8.4f} {D:>8.4f} {s:>10.4f}  {flag}")

    print()
    print("Paper's V = exp(-L/2), D = sqrt(L) gives V^2 + D^2 = exp(-L) + L")
    print("For small L: ≈ 1 + L^2/2 > 1   <-- VIOLATES bound")
    print()
    print("Correct relations for pure dephasing:")
    print("  V = |<chi_L|chi_R>|^N_gamma = exp(-Lambda_1)")
    print("  D = sqrt(1 - exp(-2 Lambda_1))")
    print("  V^2 + D^2 = exp(-2L) + (1-exp(-2L)) = 1 exactly (saturation)")
    print()


# =============================================================================
# 3. SELF-CONSISTENT OPERATING-POINT EVALUATION
# =============================================================================
def evaluate_operating_point(P_W, Delta_2pi_Hz, w, v_z=1.0, d=5e-6, N_a=1000):
    """
    Compute all derived quantities for a candidate operating point.
    Returns a dictionary of all relevant numbers.
    """
    Delta    = 2*np.pi*Delta_2pi_Hz
    zR       = np.pi*w**2/lam_L
    tau_gamma = zR/c
    tau_a    = w/v_z

    # Intensity, Rabi frequency
    I0 = 2*P_W/(np.pi*w**2)
    Omega_sq = (Gamma**2)*(I0/(2*I_sat))

    # Couplings
    g_gamma = Omega_sq*tau_gamma/(4*Delta)
    g_a     = Omega_sq*tau_a/(4*Delta)

    # Photon flux
    N_gamma = (P_W/(hbar*omega_L))*tau_a

    # Distinguishability per photon (between slits at +/- d/2 with curtain at x_c=0):
    # delta_p = (2 hbar g_gamma / w^2) * x * exp(-x^2/w^2) at x = d/2
    # Differential between slits: 2 * that
    x_slit = d/2
    delta_p_diff = 2*(2*hbar*g_gamma/w**2)*x_slit*np.exp(-x_slit**2/w**2)
    sigma_p = hbar/w
    D_1 = delta_p_diff**2/(8*sigma_p**2)
    Lambda_1 = N_gamma*D_1

    # Spontaneous emission
    Gamma_sc = Gamma*Omega_sq/(4*Delta**2)
    P_sc = Gamma_sc*tau_a

    # SNR
    SNR_ensemble = np.sqrt(N_a*Lambda_1)

    return {
        'P_W': P_W, 'Delta_GHz': Delta_2pi_Hz/1e9,
        'w_um': w*1e6, 'v_z': v_z, 'N_a': N_a,
        'tau_a_us': tau_a*1e6, 'tau_gamma_fs': tau_gamma*1e15,
        'I0_over_Isat': I0/I_sat,
        'Omega_2pi_GHz': np.sqrt(Omega_sq)/(2*np.pi)/1e9,
        'g_gamma': g_gamma, 'g_a': g_a, 'N_gamma': N_gamma,
        'D_1': D_1, 'Lambda_1': Lambda_1,
        'P_sc': P_sc, 'SNR': SNR_ensemble,
        'Delta_over_Gamma': Delta/Gamma,
    }


def search_operating_points():
    """Scan P, Delta to find a viable self-consistent operating point."""
    print("="*70)
    print("OPERATING POINT SEARCH")
    print("="*70)
    print(f"{'P':>8} {'Δ(GHz)':>8} {'w(μm)':>6} {'N_a':>8} "
          f"{'g_γ':>10} {'Λ_1':>10} {'P_sc':>10} {'SNR':>8}")
    candidates = [
        (10e-9,   10e9,  2e-6, 1.0, 1000),    # original v2 broken point
        (100e-6,  100e9, 2e-6, 1.0, 1e4),
        (300e-6,  200e9, 2e-6, 1.0, 1e4),     # CHOSEN OPERATING POINT for v3
        (1e-3,    500e9, 2e-6, 1.0, 1e4),
    ]
    for P, D, w, v, N in candidates:
        r = evaluate_operating_point(P, D, w, v, N_a=N)
        Pstr = (f"{P*1e9:.0f}nW"  if P < 1e-6 else
                f"{P*1e6:.0f}μW"  if P < 1e-3 else
                f"{P*1e3:.1f}mW")
        print(f"{Pstr:>8} {D/1e9:>8.0f} {w*1e6:>6.1f} {N:>8.0e} "
              f"{r['g_gamma']:>10.2e} {r['Lambda_1']:>10.2e} "
              f"{r['P_sc']:>10.2e} {r['SNR']:>8.1f}")

    print()
    print("CHOSEN OPERATING POINT: P=300μW, Δ=200GHz, w=2μm, N_a=1e4")
    print("  Detailed values:")
    r = evaluate_operating_point(300e-6, 200e9, 2e-6, 1.0, N_a=int(1e4))
    for k, v in r.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4g}")
        else:
            print(f"    {k}: {v}")
    print()


# =============================================================================
# 4. FIGURE 3 — operating regime in (g_gamma, N_gamma) space
# =============================================================================
def regen_figure_3(out_pdf='photon_curtain_fig3_regime.pdf',
                   out_png='photon_curtain_fig3_regime.png'):
    """
    Regenerate Figure 3 (operating regime) with the corrected operating point
    from Sec. 8 of the revised manuscript.

    D_1 derivation:
        delta_p_L - delta_p_R = 2 * (2 hbar g_gamma / w^2) * (d/2) * exp(-d^2/(4w^2))
        sigma_p = hbar / w
        => D_1 = (delta_p_diff)^2 / (8 sigma_p^2)
              = (1/2) g_gamma^2 (d/w)^2 exp(-d^2/(2w^2))
    """
    d = 5e-6
    w = 2e-6
    N_a = 1e4

    def D_1_of_g(g):
        return 0.5 * (g**2) * (d/w)**2 * np.exp(-d**2/(2*w**2))
    def Lambda_1(g, N_g):
        return N_g * D_1_of_g(g)

    g_range = np.logspace(-7, -1, 400)
    N_range = np.logspace(4, 12, 400)
    G, N = np.meshgrid(g_range, N_range)
    L1 = Lambda_1(G, N)
    NaL1 = N_a * L1

    fig, ax = plt.subplots(figsize=(8, 6.5))

    mask_strong = L1 > 1.0
    mask_weak   = NaL1 < 10
    mask_viable = (~mask_strong) & (~mask_weak)

    ax.contourf(G, N, mask_strong.astype(float),
                levels=[0.5, 1.5], colors=['#ff9999'], alpha=0.55)
    ax.contourf(G, N, mask_weak.astype(float),
                levels=[0.5, 1.5], colors=['#dddddd'], alpha=0.55)
    ax.contourf(G, N, mask_viable.astype(float),
                levels=[0.5, 1.5], colors=['#99dd99'], alpha=0.55)

    cs1 = ax.contour(G, N, L1, levels=[0.3, 1.0],
                     colors=['#cc4400', '#990000'], linewidths=1.5,
                     linestyles=['--', '-'])
    ax.clabel(cs1, fmt={0.3: r'$\Lambda_1=0.3$', 1.0: r'$\Lambda_1=1$'},
              fontsize=9)

    cs2 = ax.contour(G, N, NaL1, levels=[10],
                     colors=['#005599'], linewidths=1.5, linestyles=[':'])
    ax.clabel(cs2, fmt={10: r'$N_a\Lambda_1=10$'}, fontsize=9)

    # Operating point (corrected for v3)
    g_op = 2.2e-5
    N_op = 2.4e9
    L1_op = Lambda_1(g_op, N_op)
    ax.plot(g_op, N_op, marker='*', markersize=22,
            markerfacecolor='gold', markeredgecolor='black',
            markeredgewidth=1.5, zorder=10)
    ax.annotate('Operating point\n'
                r'$g_\gamma = 2.2 \times 10^{-5}$' + '\n'
                r'$N_\gamma^{(1)} = 2.4 \times 10^{9}$' + '\n'
                f'$\\Lambda_1 = {L1_op:.2f}$',
                xy=(g_op, N_op), xytext=(8e-6, 8e10),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.4', fc='white',
                          ec='gray', alpha=0.95))

    ax.text(8e-3, 2e11, 'Strong measurement\n(coherence destroyed)',
            fontsize=11, ha='center', color='#990000', fontweight='bold')
    ax.text(2e-7, 1e5, 'Insufficient\nstatistics',
            fontsize=11, ha='center', color='#444444', fontweight='bold')
    ax.text(2e-5, 8e6, 'Viable weak-measurement\nregime',
            fontsize=11, ha='center', color='#005500', fontweight='bold')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Per-photon coupling $g_\gamma$', fontsize=12)
    ax.set_ylabel(r'Photons per atom transit $N_\gamma^{(1)}$', fontsize=12)
    ax.set_title(r'Operating regime ($N_a = 10^4$ atoms)', fontsize=12)
    ax.set_xlim(g_range.min(), g_range.max())
    ax.set_ylim(N_range.min(), N_range.max())
    ax.grid(True, which='major', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_pdf, dpi=150, bbox_inches='tight')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {out_pdf}, {out_png}")
    print(f"  Lambda_1 at operating point: {L1_op:.4f}")
    print(f"  N_a * Lambda_1: {N_a*L1_op:.1f}")
    print(f"  In viable regime: {(L1_op < 1) and (N_a*L1_op > 10)}")


# =============================================================================
# 5. NARROW-CURTAIN LIMIT CHECK
# =============================================================================
def check_narrow_curtain_limit():
    """
    The original v2 §7 claimed:
        A(x_c) -> -sqrt(pi) w * delta'(x_a - x_c)  as w -> 0
    Verify this is mathematically correct AND note that the prefactor vanishes,
    which has consequences the paper glossed over.
    """
    print("="*70)
    print("NARROW-CURTAIN LIMIT")
    print("="*70)
    print("A(x_c) = (2(x_a-x_c)/w^2) * exp(-(x_a-x_c)^2/w^2)")
    print()
    print("In the limit w -> 0:")
    print("  exp(-(x_a-x_c)^2/w^2) ~ w * sqrt(pi) * delta(x_a-x_c)")
    print("  d/dx_a [w sqrt(pi) delta(x_a-x_c)] = w sqrt(pi) delta'(x_a-x_c)")
    print("  And A = -d/dx_a of [profile], so:")
    print("  A(x_c) -> sqrt(pi) w * delta'(x_a-x_c)  (sign: paper has -)")
    print()
    print("CRITICAL: prefactor 'sqrt(pi) * w' VANISHES as w -> 0.")
    print("=> coupling vanishes; g_gamma must scale as 1/w to compensate")
    print("=> the 'directly measures local momentum density' claim is misleading")
    print("=> v3 §6.1 now treats this honestly")
    print()


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("PHOTON CURTAIN MANUSCRIPT REVIEW — Python Computations")
    print("="*70)
    print()

    check_v2_parameters()
    check_englert_bound()
    check_narrow_curtain_limit()
    search_operating_points()
    regen_figure_3()
