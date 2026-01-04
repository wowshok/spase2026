"""
Coleman-De Luccia Instanton Solver for Cosmic Voids
====================================================

Solves the CDL instanton equations for vacuum decay in cosmic voids
with geometric suppression factor f(H) = (H/H_0)^4.

Author: Aleksandr Sergeevich Milovanov
Date: January 2026
Paper: "Hawking-Moss Vacuum Decay in Cosmic Voids: A Possible Mechanism 
        for the Hubble Tension"
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Planck mass in GeV
M_PL = 1.22e19  # GeV

# Hubble constant
H0_CMB = 67.4  # km/s/Mpc (Planck 2018)
H0_LOCAL = 73.0  # km/s/Mpc (SH0ES)

# Convert to natural units (GeV)
H0_CMB_GEV = H0_CMB * 2.13e-42  # GeV
H0_LOCAL_GEV = H0_LOCAL * 2.13e-42  # GeV

# ============================================================================
# POTENTIAL PARAMETERS
# ============================================================================

def effective_potential(phi, lambda_coupling, v, H, H0):
    """
    Effective potential with geometric suppression.
    
    V_eff = λ(φ² - v²)² · (H/H₀)⁴
    
    Parameters:
    -----------
    phi : float or array
        Scalar field value
    lambda_coupling : float
        Self-coupling constant
    v : float
        VEV (vacuum expectation value)
    H : float
        Local Hubble parameter in void
    H0 : float
        Background Hubble parameter
        
    Returns:
    --------
    V_eff : float or array
        Effective potential
    """
    # Standard double-well potential
    V_standard = lambda_coupling * (phi**2 - v**2)**2
    
    # Geometric suppression factor
    f_H = (H / H0)**4
    
    return V_standard * f_H


def potential_derivative(phi, lambda_coupling, v, H, H0):
    """
    Derivative of effective potential: dV/dφ
    
    Parameters:
    -----------
    phi : float or array
        Scalar field value
    lambda_coupling : float
        Self-coupling constant  
    v : float
        VEV
    H : float
        Local Hubble parameter
    H0 : float
        Background Hubble parameter
        
    Returns:
    --------
    dV_dphi : float or array
        Potential derivative
    """
    f_H = (H / H0)**4
    return 4 * lambda_coupling * phi * (phi**2 - v**2) * f_H


# ============================================================================
# CDL INSTANTON EQUATIONS
# ============================================================================

def cdl_equations(y, rho, lambda_coupling, v, H, H0):
    """
    CDL instanton equations in O(4) symmetry.
    
    d²φ/dρ² + (3/ρ) dφ/dρ = dV/dφ
    
    Parameters:
    -----------
    y : array [phi, phi_prime]
        State vector
    rho : float
        Radial coordinate
    lambda_coupling : float
        Coupling constant
    v : float
        VEV
    H : float
        Local Hubble parameter
    H0 : float
        Background Hubble parameter
        
    Returns:
    --------
    dydrho : array
        Derivatives [dphi/drho, d²phi/drho²]
    """
    phi, phi_prime = y
    
    # Avoid singularity at origin
    if rho < 1e-10:
        rho = 1e-10
    
    # Equation of motion
    dV_dphi = potential_derivative(phi, lambda_coupling, v, H, H0)
    phi_double_prime = -3 * phi_prime / rho + dV_dphi
    
    return [phi_prime, phi_double_prime]


def solve_cdl_instanton(lambda_coupling, v, H, H0, 
                        phi_center=None, rho_max=50.0, n_points=5000):
    """
    Solve CDL instanton equation with shooting method.
    
    Parameters:
    -----------
    lambda_coupling : float
        Self-coupling constant
    v : float
        VEV in units of M_Pl
    H : float
        Local Hubble parameter (GeV)
    H0 : float
        Background Hubble parameter (GeV)
    phi_center : float, optional
        Field value at origin. If None, use phi(0) = 0.9*v
    rho_max : float
        Maximum radial coordinate
    n_points : int
        Number of integration points
        
    Returns:
    --------
    rho : array
        Radial coordinate
    phi : array
        Field profile
    success : bool
        Whether solution is physical
    """
    # Initial conditions at origin
    if phi_center is None:
        phi_center = 0.9 * v
        
    phi_prime_center = 0.0  # Regularity at origin
    
    # Integration domain
    rho = np.linspace(0, rho_max, n_points)
    
    # Initial state
    y0 = [phi_center, phi_prime_center]
    
    # Solve ODE
    solution = odeint(cdl_equations, y0, rho, 
                     args=(lambda_coupling, v, H, H0))
    
    phi = solution[:, 0]
    phi_prime = solution[:, 1]
    
    # Check if solution reaches false vacuum
    final_phi = phi[-1]
    success = (np.abs(final_phi - v) < 0.1 * v) and (np.abs(final_phi) > 0.1 * v)
    
    return rho, phi, success


# ============================================================================
# EUCLIDEAN ACTION CALCULATION
# ============================================================================

def calculate_euclidean_action(rho, phi, lambda_coupling, v, H, H0):
    """
    Calculate Euclidean action for instanton solution.
    
    S_E = 2π² ∫ dρ ρ³ [(dφ/dρ)² + 2V(φ)]
    
    Parameters:
    -----------
    rho : array
        Radial coordinate
    phi : array
        Field profile
    lambda_coupling : float
        Coupling constant
    v : float
        VEV
    H : float
        Local Hubble parameter
    H0 : float
        Background Hubble parameter
        
    Returns:
    --------
    S_E : float
        Euclidean action
    """
    # Calculate gradient term
    dphi_drho = np.gradient(phi, rho)
    
    # Potential energy
    V = effective_potential(phi, lambda_coupling, v, H, H0)
    
    # Integrand: ρ³ [(dφ/dρ)² + 2V(φ)]
    integrand = rho**3 * (dphi_drho**2 + 2 * V)
    
    # Numerical integration (trapezoidal rule)
    S_E = 2 * np.pi**2 * np.trapz(integrand, rho)
    
    return S_E


# ============================================================================
# HAWKING-MOSS ACTION (ANALYTICAL)
# ============================================================================

def hawking_moss_action(lambda_coupling, v, H, H0):
    """
    Analytical Hawking-Moss action for comparison.
    
    S_HM ≈ (12π²)/(λv²) · (H₀/H)⁶
    
    Parameters:
    -----------
    lambda_coupling : float
        Coupling constant
    v : float
        VEV
    H : float
        Local Hubble parameter
    H0 : float
        Background Hubble parameter
        
    Returns:
    --------
    S_HM : float
        Hawking-Moss action
    """
    return (12 * np.pi**2) / (lambda_coupling * v**2) * (H0 / H)**6


# ============================================================================
# PARAMETER SCAN
# ============================================================================

def parameter_scan(lambda_range, v_range, H_values, H0,
                   save_results=True, filename='parameter_scan_results.npz'):
    """
    Scan parameter space for physical solutions.
    
    Parameters:
    -----------
    lambda_range : array
        Range of coupling constants
    v_range : array
        Range of VEV values (in M_Pl units)
    H_values : array
        Hubble parameters to test (in GeV)
    H0 : float
        Background Hubble parameter (GeV)
    save_results : bool
        Whether to save results to file
    filename : str
        Output filename
        
    Returns:
    --------
    results : dict
        Dictionary with scan results
    """
    n_lambda = len(lambda_range)
    n_v = len(v_range)
    n_H = len(H_values)
    
    # Storage arrays
    actions = np.zeros((n_lambda, n_v, n_H))
    success_flags = np.zeros((n_lambda, n_v, n_H), dtype=bool)
    
    print(f"Starting parameter scan: {n_lambda}x{n_v}x{n_H} = {n_lambda*n_v*n_H} combinations")
    
    count = 0
    total = n_lambda * n_v * n_H
    
    for i, lam in enumerate(lambda_range):
        for j, v in enumerate(v_range):
            for k, H in enumerate(H_values):
                
                # Solve instanton
                rho, phi, success = solve_cdl_instanton(lam, v, H, H0)
                
                if success:
                    # Calculate action
                    S_E = calculate_euclidean_action(rho, phi, lam, v, H, H0)
                    actions[i, j, k] = S_E
                    success_flags[i, j, k] = True
                else:
                    actions[i, j, k] = np.nan
                    success_flags[i, j, k] = False
                
                count += 1
                if count % 10 == 0:
                    print(f"Progress: {count}/{total} ({100*count/total:.1f}%)")
    
    # Prepare results
    results = {
        'lambda_range': lambda_range,
        'v_range': v_range,
        'H_values': H_values,
        'H0': H0,
        'actions': actions,
        'success': success_flags
    }
    
    if save_results:
        np.savez(filename, **results)
        print(f"Results saved to {filename}")
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_instanton_solution(rho, phi, v, lambda_coupling, H, H0, 
                           S_E=None, save_fig=False, filename='instanton.png'):
    """
    Plot instanton solution.
    
    Parameters:
    -----------
    rho : array
        Radial coordinate
    phi : array
        Field profile
    v : float
        VEV
    lambda_coupling : float
        Coupling constant
    H : float
        Local Hubble parameter
    H0 : float
        Background Hubble parameter
    S_E : float, optional
        Euclidean action
    save_fig : bool
        Whether to save figure
    filename : str
        Output filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Field profile
    ax1.plot(rho, phi / v, 'b-', linewidth=2)
    ax1.axhline(1.0, color='r', linestyle='--', label='False vacuum')
    ax1.axhline(0.0, color='g', linestyle='--', label='True vacuum')
    ax1.set_xlabel(r'$\rho$ (GeV$^{-1}$)', fontsize=12)
    ax1.set_ylabel(r'$\phi/v$', fontsize=12)
    ax1.set_title('CDL Instanton Profile', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Potential
    phi_plot = np.linspace(-1.2*v, 1.2*v, 200)
    V_plot = effective_potential(phi_plot, lambda_coupling, v, H, H0)
    
    ax2.plot(phi_plot / v, V_plot / np.max(V_plot), 'b-', linewidth=2)
    ax2.axvline(-1.0, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(1.0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel(r'$\phi/v$', fontsize=12)
    ax2.set_ylabel(r'$V_{eff}/V_{max}$', fontsize=12)
    ax2.set_title(f'Effective Potential (H/H₀ = {H/H0:.3f})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    if S_E is not None:
        fig.suptitle(f'λ = {lambda_coupling:.2e}, v = {v:.3f} M_Pl, S_E = {S_E:.2f}',
                    fontsize=12)
    
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
    print("CDL Instanton Solver - Example Calculation")
    print("=" * 70)
    
    # Physical parameters
    lambda_coupling = 1e-4  # Self-coupling
    v = 0.15  # VEV in units of M_Pl
    H0 = H0_CMB_GEV  # Background Hubble
    
    # Void Hubble parameter (enhanced by factor ~1.3)
    H_void = 1.3 * H0
    
    print(f"\nParameters:")
    print(f"  λ = {lambda_coupling:.2e}")
    print(f"  v = {v:.3f} M_Pl")
    print(f"  H₀ = {H0:.2e} GeV")
    print(f"  H_void = {H_void:.2e} GeV")
    print(f"  H_void/H₀ = {H_void/H0:.3f}")
    
    # Solve instanton
    print("\nSolving CDL instanton equation...")
    rho, phi, success = solve_cdl_instanton(lambda_coupling, v, H_void, H0)
    
    if success:
        print("✓ Physical solution found!")
        
        # Calculate action
        S_E = calculate_euclidean_action(rho, phi, lambda_coupling, v, H_void, H0)
        S_HM = hawking_moss_action(lambda_coupling, v, H_void, H0)
        
        print(f"\nResults:")
        print(f"  S_E (numerical) = {S_E:.2f}")
        print(f"  S_HM (analytical) = {S_HM:.2f}")
        print(f"  Ratio S_E/S_HM = {S_E/S_HM:.3f}")
        
        # Tunneling rate
        Gamma = np.exp(-S_E)
        print(f"  Tunneling rate Γ ~ exp(-{S_E:.2f}) = {Gamma:.2e}")
        
        # Plot solution
        plot_instanton_solution(rho, phi, v, lambda_coupling, H_void, H0, 
                               S_E=S_E, save_fig=True)
        
    else:
        print("✗ No physical solution found")
        print("  Try adjusting parameters")
    
    print("\n" + "=" * 70)
