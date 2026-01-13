import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def generate_tensile_curve(E, sigma_y, sigma_uts, epsilon_uts, sigma_f, epsilon_f, n=0.5, has_hardening=True,
                           curvature_factor=1.5, smoothing_sigma=10, num_points=5000):
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
            linear = sigma_uts + (sigma_f - sigma_uts) * (epsilon[mask_necking] - epsilon_uts) / (epsilon_f - epsilon_uts)
            perturbation = 0.05 * (sigma_uts - sigma_f) * np.sin(np.pi * (epsilon[mask_necking] - epsilon_uts) / (epsilon_f - epsilon_uts))
            sigma[mask_necking] = linear + perturbation

    # Apply Gaussian smoothing to the entire curve for smooth transitions
    sigma = gaussian_filter1d(sigma, sigma=smoothing_sigma)

    # Calculate average hardening rate (if applicable)
    if epsilon_uts > epsilon_y:
        avg_hardening_rate = (sigma_uts - sigma_y) / (epsilon_uts - epsilon_y)
    else:
        avg_hardening_rate = 0

    return epsilon, sigma, avg_hardening_rate


# ------------------------ 新增批处理功能部分 ------------------------

def parse_folder_name(folder_name):
    """
    解析文件夹名，如 E61Y661T691Ag7F264At28 或 E61.5Y660.3T690Ag6.5F250At27.5
    返回参数字典：E, sigma_y, sigma_uts, epsilon_uts, sigma_f, epsilon_f
    """
    pattern = r"E([\d\.]+)Y([\d\.]+)T([\d\.]+)Ag([\d\.]+)F([\d\.]+)At([\d\.]+)"
    match = re.match(pattern, folder_name)
    if not match:
        return None

    E = float(match.group(1))
    sigma_y = float(match.group(2))
    sigma_uts = float(match.group(3))
    epsilon_uts = float(match.group(4)) / 100  # 转为小数
    sigma_f = float(match.group(5))
    epsilon_f = float(match.group(6)) / 100    # 转为小数

    return {
        "E": E,
        "sigma_y": sigma_y,
        "sigma_uts": sigma_uts,
        "epsilon_uts": epsilon_uts,
        "sigma_f": sigma_f,
        "epsilon_f": epsilon_f
    }


if __name__ == "__main__":
    # ------- 原示例 -------
    E = 30
    sigma_y = 590
    sigma_uts = 680
    epsilon_uts = 0.17
    sigma_f = 450
    epsilon_f = 0.31
    n = 0.25
    has_hardening = True
    curvature_factor = 3
    smoothing_sigma = 1

    epsilon, sigma, avg_rate = generate_tensile_curve(
        E, sigma_y, sigma_uts, epsilon_uts, sigma_f, epsilon_f,
        n, has_hardening, curvature_factor, smoothing_sigma
    )

    print(f"Average hardening rate: {avg_rate:.2f} MPa/strain")

    plt.figure(figsize=(10, 6))
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
    plt.show()

    print("Sample data points:")
    for i in range(0, len(epsilon), len(epsilon) // 10):
        print(f"Strain: {epsilon[i]:.4f}, Stress: {sigma[i]:.2f} MPa")

    # ------- 批量自动绘图 -------
    base_dir = os.getcwd()
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(f)]

    for folder in folders:
        params = parse_folder_name(folder)
        if params is None:
            continue  # 跳过不符合命名规则的文件夹

        print(f"\nProcessing folder: {folder}")
        print(params)

        epsilon, sigma, avg_rate = generate_tensile_curve(
            params["E"], params["sigma_y"], params["sigma_uts"],
            params["epsilon_uts"], params["sigma_f"], params["epsilon_f"],
            n=0.35, has_hardening=True, curvature_factor=3, smoothing_sigma=1
        )

        plt.figure(figsize=(10, 8))
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

        save_path = os.path.join(base_dir, folder, "generated_curve.png")
        plt.savefig(save_path, dpi=600)
        plt.close()
        print(f"✅ Saved curve to: {save_path}")