import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def generate_tensile_curve(E, sigma_y, sigma_uts, epsilon_uts, sigma_f, epsilon_f, n=0.25, has_hardening=True,
                           curvature_factor=3, smoothing_sigma=1, num_points=5000):
    """
    Generate stress-strain curve based on given key points, supporting strain hardening exponent n and whether there is a hardening phase.

    Parameters:
    - E: Elastic modulus (GPa)
    - sigma_y: Yield strength (MPa)
    - sigma_uts: Ultimate tensile strength (MPa)
    - epsilon_uts: Strain at UTS (or start of necking for no hardening)
    - sigma_f: Fracture strength (MPa)
    - epsilon_f: Fracture strain
    - n: Strain hardening exponent (default 0.5, for power-law hardening)
    - has_hardening: Whether there is a hardening phase (True: has hardening; False: flat plastic to epsilon_uts, then necking)
    - curvature_factor: Factor to adjust the curvature of the necking phase (default 1.5, higher for more curvature)
    - smoothing_sigma: Sigma for Gaussian filter to smooth transitions (default 10, adjust for smoothness level)
    - num_points: Number of points (default: 5000 for smoother curve)

    Returns:
    - epsilon: Strain array
    - sigma: Corresponding stress array
    - avg_hardening_rate: Average hardening rate (MPa / strain)

    Note: Yield strain is calculated from sigma_y and E: epsilon_y = sigma_y / (E * 1000).
    If has_hardening=False, set sigma_uts = sigma_y (if not already), and create flat plastic phase from epsilon_y to epsilon_uts at sigma_y.
    Hardening phase uses power-law: sigma = sigma_y + K * epsilon_p ** n, where K is auto-matched to sigma_uts.
    Necking phase uses adjusted exponential decay for curved shape: sigma = sigma_uts * exp(-k * (epsilon - epsilon_uts)**curvature_factor), k matched to sigma_f.
    A Gaussian filter is applied to smooth transitions at yield and UTS points.
    """
    # Calculate yield strain based on yield strength
    epsilon_y = sigma_y / (E * 1000)

    if not has_hardening:
        if abs(sigma_uts - sigma_y) > 1e-6:
            print(f"Warning: For no hardening, setting sigma_uts to sigma_y ({sigma_y}).")
            sigma_uts = sigma_y
        # Do not override epsilon_uts; use it as the end of flat plastic phase

    # Calculate K for power-law hardening (or flat if K=0)
    if epsilon_uts > epsilon_y:
        delta_epsilon = epsilon_uts - epsilon_y
        delta_sigma = sigma_uts - sigma_y
        if delta_sigma == 0:  # No hardening, flat
            K = 0
            n = 0
        else:
            K = delta_sigma / (delta_epsilon ** n)
    else:
        K = 0
        n = 0

    # Generate strain range
    epsilon = np.linspace(0, epsilon_f, num_points)
    sigma = np.zeros_like(epsilon)

    # Elastic phase: ε <= epsilon_y
    mask_elastic = epsilon <= epsilon_y
    sigma[mask_elastic] = E * epsilon[mask_elastic] * 1000  # Convert to MPa

    # Hardening (or flat) phase: epsilon_y < ε <= epsilon_uts
    mask_hardening = (epsilon > epsilon_y) & (epsilon <= epsilon_uts)
    if np.any(mask_hardening):
        epsilon_p = epsilon[mask_hardening] - epsilon_y
        sigma[mask_hardening] = sigma_y + K * (epsilon_p ** n)

    # Necking phase: ε > epsilon_uts, use adjusted exponential decay for curved shape
    mask_necking = epsilon > epsilon_uts
    if np.any(mask_necking):
        if epsilon_f > epsilon_uts and sigma_f < sigma_uts:
            # Calculate adjusted decay constant k to match sigma_f at epsilon_f with curvature factor
            delta_epsilon = epsilon_f - epsilon_uts
            k = -np.log(sigma_f / sigma_uts) / (delta_epsilon ** curvature_factor)
            sigma[mask_necking] = sigma_uts * np.exp(-k * ((epsilon[mask_necking] - epsilon_uts) ** curvature_factor))
        else:
            # Fallback to linear if invalid, with slight curvature perturbation
            print("Warning: Invalid parameters for exponential necking; using linear with curvature.")
            linear = sigma_uts + (sigma_f - sigma_uts) * (epsilon[mask_necking] - epsilon_uts) / (
                        epsilon_f - epsilon_uts)
            perturbation = 0.05 * (sigma_uts - sigma_f) * np.sin(
                np.pi * (epsilon[mask_necking] - epsilon_uts) / (epsilon_f - epsilon_uts))
            sigma[mask_necking] = linear + perturbation

    # Apply Gaussian smoothing to the entire curve for smooth transitions
    sigma = gaussian_filter1d(sigma, sigma=smoothing_sigma)

    # Calculate average hardening rate (if applicable)
    if epsilon_uts > epsilon_y:
        avg_hardening_rate = (sigma_uts - sigma_y) / (epsilon_uts - epsilon_y)
    else:
        avg_hardening_rate = 0

    return epsilon, sigma, avg_hardening_rate


def plot_tensile_curve(E, sigma_y, sigma_uts, epsilon_uts, sigma_f, epsilon_f,
                       n=0.25, has_hardening=True, curvature_factor=3, smoothing_sigma=1, num_points=5000):
    """
    Generate and plot the tensile curve, returning the matplotlib figure object.

    Parameters:
    - Same as generate_tensile_curve.

    Returns:
    - fig: Matplotlib figure object
    """
    epsilon, sigma, avg_rate = generate_tensile_curve(
        E, sigma_y, sigma_uts, epsilon_uts, sigma_f, epsilon_f,
        n, has_hardening, curvature_factor, smoothing_sigma, num_points
    )

    fig = plt.figure(figsize=(10, 6))
    plt.plot(epsilon * 100, sigma, color='gray', linewidth=3)
    plt.xlim(0, 50)
    plt.ylim(0, 1000)
    plt.xticks(np.arange(0, 51, 5))
    plt.yticks(np.arange(0, 1001, 200))
    plt.xlabel('Engineering strain (%)', fontsize=26)
    plt.ylabel('Engineering stress (MPa)', fontsize=26)
    plt.tick_params(axis='both', direction='in', labelsize=24, width=2)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()

    return fig
