# Python Code for "Hawking-Moss Vacuum Decay in Cosmic Voids"

**Author:** Aleksandr Sergeevich Milovanov  
**Date:** January 2026  
**Paper:** "Hawking-Moss Vacuum Decay in Cosmic Voids: A Possible Mechanism for the Hubble Tension"

---

## Overview

This package contains Python implementations of the numerical methods used in the paper, including:

1. **CDL Instanton Solver** - Numerical solutions of Coleman-De Luccia equations
2. **Void Evolution Model** - Temporal evolution of cosmic voids
3. **Hubble Tension Analysis** - Mock data analysis and statistical testing

---

## Files

### Main Scripts

```
cdl_instanton_solver.py      - CDL/Hawking-Moss instanton calculations
void_evolution_model.py      - Void density and Hubble parameter evolution  
hubble_tension_analysis.py   - Statistical analysis with mock SNe data
```

### Documentation

```
README.md           - This file
requirements.txt    - Python package dependencies
```

---

## Installation

### Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy scipy matplotlib
```

---

## Usage

### 1. CDL Instanton Solver

Solves the instanton equations and calculates Euclidean action:

```python
python cdl_instanton_solver.py
```

**Key Functions:**

```python
from cdl_instanton_solver import solve_cdl_instanton, calculate_euclidean_action

# Solve instanton
lambda_coupling = 1e-4
v = 0.15  # M_Pl units
H_void = 1.3 * H0_CMB_GEV
H0 = H0_CMB_GEV

rho, phi, success = solve_cdl_instanton(lambda_coupling, v, H_void, H0)

# Calculate action
if success:
    S_E = calculate_euclidean_action(rho, phi, lambda_coupling, v, H_void, H0)
    print(f"Euclidean action: {S_E:.2f}")
```

**Outputs:**
- Instanton field profile φ(ρ)
- Euclidean action S_E
- Tunneling rate Γ ~ exp(-S_E)
- Comparison with Hawking-Moss analytical formula

**Figures:**
- `instanton.png` - Field profile and effective potential

---

### 2. Void Evolution Model

Models void evolution from formation to transition:

```python
python void_evolution_model.py
```

**Key Functions:**

```python
from void_evolution_model import solve_void_evolution, transition_time_prediction

# Evolve void
R_void = 200  # Mpc
t, delta = solve_void_evolution(R_void)

# Predict transition
t_trans = transition_time_prediction(R_void)
print(f"Transition time: {t_trans:.1f} Gyr")
```

**Outputs:**
- Density contrast evolution δ(t)
- Local Hubble parameter H_local(t)
- Transition time predictions
- Critical radius calculations

**Figures:**
- `void_evolution.png` - Evolution trajectories
- `transition_predictions.png` - Critical radius vs λ

---

### 3. Hubble Tension Analysis

Analyzes void-H₀ correlations with mock data:

```python
python hubble_tension_analysis.py
```

**Key Functions:**

```python
from hubble_tension_analysis import (generate_void_catalog, 
                                     generate_sne_catalog,
                                     simulate_local_h0,
                                     correlation_analysis)

# Generate mock data
voids = generate_void_catalog(n_voids=50)
sne = generate_sne_catalog(n_sne=1000)

# Simulate H₀ with void enhancement
H0_local, d_void = simulate_local_h0(sne, voids, 
                                     enhancement_amplitude=4.0,
                                     scale_length=150.0)

# Statistical analysis
stats = correlation_analysis(d_void, H0_local)
print(f"Correlation: r = {stats['r_pearson']:.3f}, p = {stats['p_pearson']:.2e}")
```

**Outputs:**
- Pearson/Spearman correlations
- Binned H₀ analysis (near/mid/far from voids)
- Fraction of Hubble tension explained

**Figures:**
- `hubble_analysis.png` - Comprehensive analysis plot

---

## Physical Parameters

### Constants Used

```python
# Planck mass
M_PL = 1.22e19 GeV

# Hubble constant
H0_CMB = 67.4 km/s/Mpc   # Planck 2018
H0_LOCAL = 73.0 km/s/Mpc  # SH0ES

# Matter density
OMEGA_M = 0.315

# Universe age
T_UNIVERSE = 13.8 Gyr
```

### Optimal Parameters (from paper)

```python
lambda_coupling = 1e-4      # Self-coupling
v = 0.15                    # VEV (M_Pl units)
R_crit ≈ 194 Mpc           # Critical void radius
t_trans ≈ 16.7 Gyr         # Transition time
```

---

## Code Structure

### CDL Instanton Solver

```
cdl_instanton_solver.py
├── Physical constants
├── effective_potential()          - V_eff with geometric suppression
├── potential_derivative()          - dV/dφ
├── cdl_equations()                 - ODE system
├── solve_cdl_instanton()           - Main solver (shooting method)
├── calculate_euclidean_action()    - S_E calculation
├── hawking_moss_action()           - Analytical S_HM
├── parameter_scan()                - Grid search
└── plot_instanton_solution()       - Visualization
```

### Void Evolution Model

```
void_evolution_model.py
├── Cosmological parameters
├── delta_evolution()               - Density evolution ODE
├── solve_void_evolution()          - Main solver
├── local_hubble_parameter()        - H_local(δ)
├── critical_radius_prediction()    - R_crit from S_E ~ 1
├── transition_time_prediction()    - When δ → δ_crit
├── plot_void_evolution()           - Multi-panel plots
└── plot_transition_predictions()   - λ dependence
```

### Hubble Tension Analysis

```
hubble_tension_analysis.py
├── Mock data generation
├── generate_void_catalog()         - Random void distribution
├── generate_sne_catalog()          - Random SNe positions
├── calculate_void_proximity()      - Distance to nearest void
├── simulate_local_h0()             - H₀ with void enhancement
├── correlation_analysis()          - Pearson/Spearman tests
└── plot_hubble_tension_analysis()  - Comprehensive plots
```

---

## Example Outputs

### CDL Instanton

```
===============================================================
CDL Instanton Solver - Example Calculation
===============================================================

Parameters:
  λ = 1.00e-04
  v = 0.150 M_Pl
  H₀ = 1.43e-42 GeV
  H_void = 1.86e-42 GeV
  H_void/H₀ = 1.300

Solving CDL instanton equation...
✓ Physical solution found!

Results:
  S_E (numerical) = 0.87
  S_HM (analytical) = 0.92
  Ratio S_E/S_HM = 0.946
  Tunneling rate Γ ~ exp(-0.87) = 4.2e-01
```

### Void Evolution

```
Transition time predictions:
  R =  50 Mpc → No transition within Hubble time
  R = 100 Mpc → No transition within Hubble time
  R = 150 Mpc → t_trans = 18.2 Gyr
  R = 200 Mpc → t_trans = 16.7 Gyr
  R = 250 Mpc → t_trans = 15.3 Gyr
```

### Hubble Tension

```
Results:
  Pearson correlation: r = -0.782, p = 3.4e-12
  Spearman correlation: r = -0.771, p = 8.1e-12

  H₀ near voids: 70.9 ± 2.1 km/s/Mpc
  H₀ far from voids: 67.0 ± 1.8 km/s/Mpc
  Difference: ΔH₀ = 3.9 km/s/Mpc

  Hubble tension explained: 65%
```

---

## Reproducibility

All scripts use fixed random seeds where applicable for reproducibility:

```python
np.random.seed(42)
```

Results may vary slightly due to numerical precision but should be consistent within uncertainties.

---

## Performance Notes

### Computational Cost

| Script | Typical Runtime | Memory |
|--------|----------------|--------|
| CDL solver (single) | ~1 second | <100 MB |
| Parameter scan (400 points) | ~10 minutes | ~500 MB |
| Void evolution | ~5 seconds | <100 MB |
| Hubble analysis (1000 SNe) | ~10 seconds | <200 MB |

### Optimization Tips

For parameter scans:

```python
# Use parallel processing
from multiprocessing import Pool

def scan_parallel(params):
    # Your scan function
    pass

with Pool(4) as p:
    results = p.map(scan_parallel, param_list)
```

---

## Extending the Code

### Custom Potential

Modify `effective_potential()` in `cdl_instanton_solver.py`:

```python
def effective_potential(phi, lambda_coupling, v, H, H0):
    # Your custom potential here
    V = ...  # e.g., polynomial, exponential, etc.
    f_H = (H / H0)**4  # Keep geometric suppression
    return V * f_H
```

### Different Void Profiles

In `void_evolution_model.py`:

```python
def delta_evolution(delta, t, R_void, profile='exponential'):
    if profile == 'exponential':
        # Current implementation
        ...
    elif profile == 'gaussian':
        # Gaussian profile
        ...
```

---

## Citation

If you use this code, please cite:

```
Milovanov, A.S. (2026). Hawking-Moss Vacuum Decay in Cosmic Voids: 
A Possible Mechanism for the Hubble Tension. 
arXiv:XXXX.XXXXX [astro-ph.CO]
```

---

## License

Creative Commons Attribution 4.0 International (CC BY 4.0)

You are free to:
- Share and redistribute
- Adapt and build upon

With attribution to the author.

---

## Contact

**Author:** Aleksandr Sergeevich Milovanov  
**Email:** mrneefu@gmail.com  
**Status:** Independent Researcher

For questions, bug reports, or collaboration inquiries, please contact via email.

---

## Acknowledgments

This code was developed with assistance from Anthropic's Claude AI for numerical methods implementation and documentation.

Scientific computing libraries:
- NumPy (Harris et al. 2020)
- SciPy (Virtanen et al. 2020)
- Matplotlib (Hunter 2007)

---

## Version History

**v1.0 (2026-01-04):**
- Initial release
- CDL instanton solver
- Void evolution model
- Hubble tension analysis

---

## References

See main paper for complete references to:
- Coleman-De Luccia formalism
- Hawking-Moss transitions
- Cosmic void catalogs (SDSS, 2MRS)
- Hubble constant measurements (Planck, SH0ES)
