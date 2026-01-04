"""
Void Evolution Model
====================

Models the temporal evolution of cosmic voids from formation to 
critical transition, including density contrast, Hubble parameter 
evolution, and transition predictions.

Author: Aleksandr Sergeevich Milovanov
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

# ============================================================================
# COSMOLOGICAL PARAMETERS
# ============================================================================

# Hubble constant
H0_CMB = 67.4  # km/s/Mpc
H0_LOCAL = 73.0  # km/s/Mpc

# Matter density parameter
OMEGA_M = 0.315

# Dark energy density
OMEGA_LAMBDA = 1 - OMEGA_M

# Age of universe (Gyr)
T_UNIVERSE = 13.8

# ============================================================================
# VOID EVOLUTION EQUATIONS
# ============================================================================

def delta_evolution(delta, t, R_void, Omega_m, Omega_lambda):
    """
    Evolution equation for void density contrast.
    
    dδ/dt ∝ -Ω_m·δ / (1 + δ)
    
    Parameters:
    -----------
    delta : float
        Density contrast (negative for voids)
    t : float
        Time (Gyr)
    R_void : float
        Void radius (Mpc)
    Omega_m : float
        Matter density parameter
    Omega_lambda : float
        Dark energy density parameter
        
    Returns:
    --------
    d_delta_dt : float
        Time derivative of density contrast
    """
    # Scale factor evolution
    a = ((Omega_m / Omega_lambda) * np.sinh(1.5 * np.sqrt(Omega_lambda) * t))**(2/3)
    
    # Hubble parameter
    H = H0_CMB * np.sqrt(Omega_m / a**3 + Omega_lambda)
    
    # Void growth rate (simplified)
    # Larger voids evolve slower
    growth_rate = -H * Omega_m * delta / (1 + delta) / (1 + R_void / 100.0)
    
    return growth_rate


def solve_void_evolution(R_void, delta_initial=-0.3, t_max=T_UNIVERSE, n_points=1000):
    """
    Solve void evolution from formation to present.
    
    Parameters:
    -----------
    R_void : float
        Void radius (Mpc)
    delta_initial : float
        Initial density contrast
    t_max : float
        Final time (Gyr)
    n_points : int
        Number of time points
        
    Returns:
    --------
    t : array
        Time (Gyr)
    delta : array
        Density contrast evolution
    """
    t = np.linspace(0.1, t_max, n_points)  # Start at t=0.1 to avoid singularity
    
    delta = odeint(delta_evolution, delta_initial, t, 
                  args=(R_void, OMEGA_M, OMEGA_LAMBDA))
    
    return t, delta.flatten()


# ============================================================================
# HUBBLE PARAMETER IN VOIDS
# ============================================================================

def local_hubble_parameter(delta, H0=H0_CMB):
    """
    Local Hubble parameter in void.
    
    H_local = H₀ · (1 - δ/3)  for small δ
    
    For large voids: H_local = H₀ · (1 + |δ|)^(1/3)
    
    Parameters:
    -----------
    delta : float or array
        Density contrast
    H0 : float
        Background Hubble constant
        
    Returns:
    --------
    H_local : float or array
        Local Hubble parameter
    """
    # Linear approximation for small underdensity
    if np.abs(delta) < 0.5:
        return H0 * (1 - delta / 3)
    
    # Non-linear regime
    return H0 * (1 + np.abs(delta))**(1/3)


# ============================================================================
# TRANSITION PREDICTION
# ============================================================================

def critical_radius_prediction(lambda_coupling, v, H0):
    """
    Predict critical void radius for quantum transition.
    
    From condition S_E ≲ 1:
    R_crit ≈ (λv²/12π²)^(1/6) · c/H₀
    
    Parameters:
    -----------
    lambda_coupling : float
        Self-coupling constant
    v : float
        VEV (in M_Pl units)
    H0 : float
        Hubble constant (km/s/Mpc)
        
    Returns:
    --------
    R_crit : float
        Critical radius (Mpc)
    """
    # Speed of light
    c = 3e5  # km/s
    
    # Critical radius
    R_crit = (lambda_coupling * v**2 / (12 * np.pi**2))**(1/6) * (c / H0)
    
    return R_crit


def transition_time_prediction(R_void, delta_crit=-0.8):
    """
    Predict when void reaches critical density for transition.
    
    Parameters:
    -----------
    R_void : float
        Void radius (Mpc)
    delta_crit : float
        Critical density contrast
        
    Returns:
    --------
    t_transition : float
        Transition time (Gyr)
    """
    # Solve for time when delta = delta_crit
    t, delta = solve_void_evolution(R_void)
    
    # Find crossing point
    idx = np.where(delta <= delta_crit)[0]
    
    if len(idx) > 0:
        return t[idx[0]]
    else:
        return np.inf  # Never reaches critical density


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_void_evolution(R_values, save_fig=False, filename='void_evolution.png'):
    """
    Plot void evolution for different radii.
    
    Parameters:
    -----------
    R_values : array
        Void radii to plot (Mpc)
    save_fig : bool
        Whether to save figure
    filename : str
        Output filename
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(R_values)))
    
    for R, color in zip(R_values, colors):
        t, delta = solve_void_evolution(R)
        H_local = local_hubble_parameter(delta)
        
        # Density contrast evolution
        ax1.plot(t, delta, color=color, linewidth=2, label=f'R = {R} Mpc')
        
        # Hubble parameter evolution
        ax2.plot(t, H_local, color=color, linewidth=2)
        
        # H/H₀ ratio
        ax3.plot(t, H_local / H0_CMB, color=color, linewidth=2)
        
        # Phase space (δ vs H/H₀)
        ax4.plot(delta, H_local / H0_CMB, color=color, linewidth=2)
    
    # Density contrast
    ax1.axhline(-0.8, color='r', linestyle='--', alpha=0.5, label='Critical δ')
    ax1.set_xlabel('Time (Gyr)', fontsize=12)
    ax1.set_ylabel('Density Contrast δ', fontsize=12)
    ax1.set_title('Void Density Evolution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Hubble parameter
    ax2.axhline(H0_CMB, color='k', linestyle='--', alpha=0.5, label='H₀ (CMB)')
    ax2.axhline(H0_LOCAL, color='r', linestyle='--', alpha=0.5, label='H₀ (local)')
    ax2.set_xlabel('Time (Gyr)', fontsize=12)
    ax2.set_ylabel('H_local (km/s/Mpc)', fontsize=12)
    ax2.set_title('Local Hubble Parameter', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # H/H₀ ratio
    ax3.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax3.axhline(H0_LOCAL / H0_CMB, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (Gyr)', fontsize=12)
    ax3.set_ylabel('H_local / H₀', fontsize=12)
    ax3.set_title('Hubble Enhancement Factor', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Phase space
    ax4.axvline(-0.8, color='r', linestyle='--', alpha=0.5)
    ax4.axhline(H0_LOCAL / H0_CMB, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Density Contrast δ', fontsize=12)
    ax4.set_ylabel('H_local / H₀', fontsize=12)
    ax4.set_title('Phase Space Trajectory', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")
    
    plt.show()


def plot_transition_predictions(lambda_range, v_value=0.15,
                                save_fig=False, filename='transition_predictions.png'):
    """
    Plot critical radius and transition time predictions.
    
    Parameters:
    -----------
    lambda_range : array
        Range of coupling constants
    v_value : float
        VEV value (M_Pl units)
    save_fig : bool
        Whether to save figure
    filename : str
        Output filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    R_crit = [critical_radius_prediction(lam, v_value, H0_CMB) for lam in lambda_range]
    t_trans = [transition_time_prediction(R) for R in R_crit]
    
    # Critical radius vs λ
    ax1.loglog(lambda_range, R_crit, 'b-', linewidth=2)
    ax1.axhline(180, color='r', linestyle='--', label='Typical supervoid')
    ax1.set_xlabel(r'Coupling Constant $\lambda$', fontsize=12)
    ax1.set_ylabel('Critical Radius (Mpc)', fontsize=12)
    ax1.set_title('Critical Void Radius for Transition', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Transition time vs λ
    ax2.semilogx(lambda_range, t_trans, 'b-', linewidth=2)
    ax2.axhline(T_UNIVERSE, color='r', linestyle='--', label='Age of Universe')
    ax2.set_xlabel(r'Coupling Constant $\lambda$', fontsize=12)
    ax2.set_ylabel('Transition Time (Gyr)', fontsize=12)
    ax2.set_title('Predicted Transition Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")
    
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("Void Evolution Model - Example Calculations")
    print("=" * 70)
    
    # Define void sizes to study
    R_values = np.array([50, 100, 150, 200, 250])  # Mpc
    
    print(f"\nVoid radii: {R_values} Mpc")
    
    # Calculate transition times
    print("\nTransition time predictions:")
    for R in R_values:
        t_trans = transition_time_prediction(R)
        if np.isfinite(t_trans):
            print(f"  R = {R:3d} Mpc → t_trans = {t_trans:.1f} Gyr")
        else:
            print(f"  R = {R:3d} Mpc → No transition within Hubble time")
    
    # Plot evolution
    print("\nGenerating void evolution plot...")
    plot_void_evolution(R_values, save_fig=True)
    
    # Critical radius predictions
    print("\nCritical radius predictions for different λ:")
    lambda_range = np.logspace(-5, -3, 20)
    
    for lam in [1e-5, 1e-4, 1e-3]:
        R_crit = critical_radius_prediction(lam, 0.15, H0_CMB)
        print(f"  λ = {lam:.0e} → R_crit = {R_crit:.0f} Mpc")
    
    # Plot transition predictions
    print("\nGenerating transition prediction plot...")
    plot_transition_predictions(lambda_range, save_fig=True)
    
    print("\n" + "=" * 70)
