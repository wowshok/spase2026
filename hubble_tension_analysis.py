"""
Hubble Tension Analysis with Mock Data
=======================================

Analyzes correlation between void proximity and local H₀ measurements
using mock SNe Ia data to test the void-based Hubble tension mechanism.

Author: Aleksandr Sergeevich Milovanov
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit

# ============================================================================
# COSMOLOGICAL PARAMETERS
# ============================================================================

H0_CMB = 67.4  # km/s/Mpc (Planck)
H0_LOCAL = 73.0  # km/s/Mpc (SH0ES)
HUBBLE_TENSION = H0_LOCAL - H0_CMB

# ============================================================================
# MOCK DATA GENERATION
# ============================================================================

def generate_void_catalog(n_voids=100, R_min=50, R_max=300):
    """
    Generate mock void catalog.
    
    Parameters:
    -----------
    n_voids : int
        Number of voids
    R_min, R_max : float
        Radius range (Mpc)
        
    Returns:
    --------
    voids : dict
        Void catalog with positions and radii
    """
    # Random void positions (in comoving coordinates)
    positions = np.random.randn(n_voids, 3) * 500  # Mpc
    
    # Void radii (power-law distribution favoring larger voids)
    alpha = 2.5  # Power-law index
    u = np.random.random(n_voids)
    radii = R_min * (R_max/R_min) ** u
    
    voids = {
        'positions': positions,
        'radii': radii,
        'n_voids': n_voids
    }
    
    return voids


def generate_sne_catalog(n_sne=1000, z_min=0.01, z_max=0.15):
    """
    Generate mock SNe Ia catalog.
    
    Parameters:
    -----------
    n_sne : int
        Number of supernovae
    z_min, z_max : float
        Redshift range
        
    Returns:
    --------
    sne : dict
        SNe catalog with positions and redshifts
    """
    # Random SNe positions
    positions = np.random.randn(n_sne, 3) * 500  # Mpc
    
    # Distances from positions
    distances = np.linalg.norm(positions, axis=1)
    
    # Redshifts (approximate)
    c = 3e5  # km/s
    redshifts = H0_CMB * distances / c
    
    # Filter to desired range
    mask = (redshifts >= z_min) & (redshifts <= z_max)
    
    sne = {
        'positions': positions[mask],
        'redshifts': redshifts[mask],
        'distances': distances[mask],
        'n_sne': np.sum(mask)
    }
    
    return sne


def calculate_void_proximity(sne_positions, void_catalog):
    """
    Calculate minimum distance to nearest void for each SN.
    
    Parameters:
    -----------
    sne_positions : array (n_sne, 3)
        SN positions
    void_catalog : dict
        Void catalog
        
    Returns:
    --------
    d_min : array
        Minimum distance to void edge (Mpc)
    """
    n_sne = len(sne_positions)
    n_voids = void_catalog['n_voids']
    
    d_min = np.zeros(n_sne)
    
    for i, sn_pos in enumerate(sne_positions):
        # Distance to all void centers
        void_pos = void_catalog['positions']
        void_r = void_catalog['radii']
        
        distances_to_centers = np.linalg.norm(void_pos - sn_pos, axis=1)
        
        # Distance to void edge (negative if inside)
        distances_to_edges = distances_to_centers - void_r
        
        # Minimum distance
        d_min[i] = np.min(np.abs(distances_to_edges))
    
    return d_min


# ============================================================================
# MOCK H₀ MEASUREMENTS
# ============================================================================

def simulate_local_h0(sne_catalog, void_catalog, 
                     enhancement_amplitude=4.0,  # km/s/Mpc
                     scale_length=150.0):  # Mpc
    """
    Simulate local H₀ measurements with void-induced enhancement.
    
    H_local = H₀_CMB + A · exp(-d/L)
    
    Parameters:
    -----------
    sne_catalog : dict
        SNe catalog
    void_catalog : dict
        Void catalog
    enhancement_amplitude : float
        Maximum H₀ enhancement (km/s/Mpc)
    scale_length : float
        Characteristic length scale (Mpc)
        
    Returns:
    --------
    H0_local : array
        Local H₀ measurements
    """
    # Calculate void proximity
    d_void = calculate_void_proximity(sne_catalog['positions'], void_catalog)
    
    # Enhancement decreases with distance from void
    enhancement = enhancement_amplitude * np.exp(-d_void / scale_length)
    
    # Base Hubble constant + enhancement
    H0_base = H0_CMB
    H0_local = H0_base + enhancement
    
    # Add measurement noise
    noise = np.random.normal(0, 1.5, len(H0_local))  # 1.5 km/s/Mpc uncertainty
    
    return H0_local + noise, d_void


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def correlation_analysis(d_void, H0_measurements):
    """
    Perform correlation analysis.
    
    Parameters:
    -----------
    d_void : array
        Distance to nearest void (Mpc)
    H0_measurements : array
        Local H₀ measurements (km/s/Mpc)
        
    Returns:
    --------
    stats : dict
        Statistical results
    """
    # Pearson correlation
    r_pearson, p_pearson = pearsonr(-d_void, H0_measurements)  # negative: closer → higher H₀
    
    # Spearman correlation
    r_spearman, p_spearman = spearmanr(-d_void, H0_measurements)
    
    # Binned analysis
    bins = np.percentile(d_void, [0, 33, 67, 100])
    H0_near = H0_measurements[d_void < bins[1]]
    H0_mid = H0_measurements[(d_void >= bins[1]) & (d_void < bins[2])]
    H0_far = H0_measurements[d_void >= bins[2]]
    
    stats = {
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'r_spearman': r_spearman,
        'p_spearman': p_spearman,
        'H0_near_mean': np.mean(H0_near),
        'H0_near_std': np.std(H0_near),
        'H0_mid_mean': np.mean(H0_mid),
        'H0_mid_std': np.std(H0_mid),
        'H0_far_mean': np.mean(H0_far),
        'H0_far_std': np.std(H0_far),
        'bins': bins
    }
    
    return stats


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_hubble_tension_analysis(d_void, H0_measurements, stats,
                                 save_fig=False, filename='hubble_analysis.png'):
    """
    Plot Hubble tension analysis results.
    
    Parameters:
    -----------
    d_void : array
        Distance to voids
    H0_measurements : array
        H₀ measurements
    stats : dict
        Statistical results
    save_fig : bool
        Whether to save figure
    filename : str
        Output filename
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Scatter plot
    ax1 = fig.add_subplot(gs[0, :2])
    scatter = ax1.scatter(d_void, H0_measurements, 
                         c=H0_measurements, cmap='RdYlBu_r',
                         alpha=0.6, s=30)
    
    # Fit line
    def exp_fit(d, A, L):
        return H0_CMB + A * np.exp(-d / L)
    
    try:
        popt, _ = curve_fit(exp_fit, d_void, H0_measurements, 
                           p0=[4.0, 150.0], maxfev=5000)
        d_fit = np.linspace(0, np.max(d_void), 100)
        ax1.plot(d_fit, exp_fit(d_fit, *popt), 'r-', linewidth=2,
                label=f'Fit: H₀ = {H0_CMB:.1f} + {popt[0]:.1f}·exp(-d/{popt[1]:.0f})')
    except:
        pass
    
    ax1.axhline(H0_CMB, color='b', linestyle='--', label=f'H₀ (CMB) = {H0_CMB} km/s/Mpc')
    ax1.axhline(H0_LOCAL, color='r', linestyle='--', label=f'H₀ (local) = {H0_LOCAL} km/s/Mpc')
    
    ax1.set_xlabel('Distance to Nearest Void (Mpc)', fontsize=12)
    ax1.set_ylabel('Local H₀ (km/s/Mpc)', fontsize=12)
    ax1.set_title(f'Hubble Constant vs Void Proximity\n(r = {stats["r_pearson"]:.3f}, p = {stats["p_pearson"]:.2e})',
                 fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('H₀ (km/s/Mpc)', fontsize=10)
    
    # Binned analysis
    ax2 = fig.add_subplot(gs[0, 2])
    
    bins_centers = ['Near\n(<33%)', 'Mid\n(33-67%)', 'Far\n(>67%)']
    H0_means = [stats['H0_near_mean'], stats['H0_mid_mean'], stats['H0_far_mean']]
    H0_stds = [stats['H0_near_std'], stats['H0_mid_std'], stats['H0_far_std']]
    
    colors_bins = ['red', 'orange', 'blue']
    ax2.bar(bins_centers, H0_means, yerr=H0_stds, 
           color=colors_bins, alpha=0.7, capsize=5)
    
    ax2.axhline(H0_CMB, color='b', linestyle='--', linewidth=2)
    ax2.axhline(H0_LOCAL, color='r', linestyle='--', linewidth=2)
    
    ax2.set_ylabel('Mean H₀ (km/s/Mpc)', fontsize=12)
    ax2.set_title('Binned Analysis', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(H0_measurements, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax3.axvline(H0_CMB, color='b', linestyle='--', linewidth=2, label='CMB')
    ax3.axvline(H0_LOCAL, color='r', linestyle='--', linewidth=2, label='SH0ES')
    ax3.axvline(np.mean(H0_measurements), color='g', linestyle='-', linewidth=2, 
               label=f'Mean = {np.mean(H0_measurements):.1f}')
    
    ax3.set_xlabel('H₀ (km/s/Mpc)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('H₀ Distribution', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Distance distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(d_void, bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
    ax4.axvline(np.median(d_void), color='r', linestyle='--', linewidth=2,
               label=f'Median = {np.median(d_void):.0f} Mpc')
    
    ax4.set_xlabel('Distance to Void (Mpc)', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Void Proximity Distribution', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Statistics summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    stats_text = f"""
    Statistical Summary
    ═══════════════════
    
    Correlation:
      Pearson r = {stats['r_pearson']:.3f}
      p-value = {stats['p_pearson']:.2e}
    
    Binned Analysis:
      Near voids:
        H₀ = {stats['H0_near_mean']:.1f} ± {stats['H0_near_std']:.1f}
      
      Far from voids:
        H₀ = {stats['H0_far_mean']:.1f} ± {stats['H0_far_std']:.1f}
    
      Difference:
        ΔH₀ = {stats['H0_near_mean'] - stats['H0_far_mean']:.1f} km/s/Mpc
    
    Hubble Tension:
      Observed: {HUBBLE_TENSION:.1f} km/s/Mpc
      Explained: {100*(stats['H0_near_mean'] - stats['H0_far_mean'])/HUBBLE_TENSION:.0f}%
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Mock Hubble Tension Analysis: Void Proximity Effect', 
                fontsize=16, fontweight='bold')
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")
    
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("Hubble Tension Analysis - Mock Data Study")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate catalogs
    print("\nGenerating mock catalogs...")
    void_catalog = generate_void_catalog(n_voids=50, R_min=80, R_max=300)
    sne_catalog = generate_sne_catalog(n_sne=1000, z_min=0.01, z_max=0.15)
    
    print(f"  Voids: {void_catalog['n_voids']}")
    print(f"  SNe: {sne_catalog['n_sne']}")
    
    # Simulate H₀ measurements
    print("\nSimulating H₀ measurements with void enhancement...")
    H0_measurements, d_void = simulate_local_h0(sne_catalog, void_catalog,
                                               enhancement_amplitude=4.0,
                                               scale_length=150.0)
    
    print(f"  Mean H₀ = {np.mean(H0_measurements):.2f} ± {np.std(H0_measurements):.2f} km/s/Mpc")
    
    # Statistical analysis
    print("\nPerforming correlation analysis...")
    stats = correlation_analysis(d_void, H0_measurements)
    
    print(f"\nResults:")
    print(f"  Pearson correlation: r = {stats['r_pearson']:.3f}, p = {stats['p_pearson']:.2e}")
    print(f"  Spearman correlation: r = {stats['r_spearman']:.3f}, p = {stats['p_spearman']:.2e}")
    print(f"\n  H₀ near voids: {stats['H0_near_mean']:.1f} ± {stats['H0_near_std']:.1f} km/s/Mpc")
    print(f"  H₀ far from voids: {stats['H0_far_mean']:.1f} ± {stats['H0_far_std']:.1f} km/s/Mpc")
    print(f"  Difference: ΔH₀ = {stats['H0_near_mean'] - stats['H0_far_mean']:.1f} km/s/Mpc")
    
    explained_fraction = (stats['H0_near_mean'] - stats['H0_far_mean']) / HUBBLE_TENSION
    print(f"\n  Hubble tension explained: {100*explained_fraction:.0f}%")
    
    # Visualization
    print("\nGenerating plots...")
    plot_hubble_tension_analysis(d_void, H0_measurements, stats, save_fig=True)
    
    print("\n" + "=" * 70)
