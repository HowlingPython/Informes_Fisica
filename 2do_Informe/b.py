import os
import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data skipping first two rows.
    
    Args:
        file_path: Path to data file.
    
    Returns:
        DataFrame with columns ['t', 'x', 'y'].
    """
    return pd.read_csv(file_path, skiprows=2, header=None, names=['t', 'x', 'y'], delimiter=',')

def calculate_height(signal: np.ndarray) -> float:
    """
    Calculate peak height threshold for peak detection.
    
    Args:
        signal: 1D numpy array of signal values.
    
    Returns:
        Height threshold as float.
    """
    return 0.5 * (signal.max() - signal.min()) + signal.min()

def find_signal_peaks(signal: np.ndarray, height: float, distance: int) -> np.ndarray:
    """
    Find indices of peaks in the signal.
    
    Args:
        signal: 1D numpy array.
        height: Minimum height of peaks.
        distance: Minimum distance between peaks.
    
    Returns:
        Indices of peaks in the signal.
    """
    peaks, _ = find_peaks(signal, height=height, distance=distance)
    return peaks

def calculate_omega_and_period(t: np.ndarray, peaks: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculate angular frequency omega and period T with uncertainties.
    
    Args:
        t: Time array.
        peaks: Indices of detected peaks.
    
    Returns:
        omega, sigma_omega, period, sigma_period
    """
    if len(peaks) < 2:
        return np.nan, np.nan, np.nan, np.nan

    t_peaks = t[peaks]
    intervals = np.diff(t_peaks)
    n = len(intervals)

    period_mean = intervals.mean()
    sigma_period = intervals.std(ddof=1) / np.sqrt(n)

    omega = 2 * np.pi / period_mean
    sigma_omega = (2 * np.pi / period_mean**2) * sigma_period

    return omega, sigma_omega, period_mean, sigma_period

def get_initial_amplitude(signal: np.ndarray, peaks: np.ndarray) -> float:
    """
    Get initial oscillation amplitude (absolute value of first peak).
    
    Args:
        signal: Signal array.
        peaks: Indices of peaks.
    
    Returns:
        Amplitude at first peak or np.nan if none.
    """
    if len(peaks) == 0:
        return np.nan
    return abs(signal[peaks[0]])

def amplitude_over_time(signal: np.ndarray, t: np.ndarray, peaks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract amplitudes and their times from signal peaks.
    
    Args:
        signal: Signal array.
        t: Time array.
        peaks: Indices of peaks.
    
    Returns:
        times and amplitudes arrays.
    """
    return t[peaks], np.abs(signal[peaks])

def plot_multiple_signals(df_list: List[pd.DataFrame], filenames: List[str], height: float, distance: int) -> None:
    """
    Plot all signals with peaks detected.
    
    Args:
        df_list: List of dataframes each containing 't' and 'x'.
        filenames: List of filenames for titles.
        height: Peak detection height threshold.
        distance: Peak detection minimum distance.
    """
    for df, fname in zip(df_list, filenames):
        t = df['t'].to_numpy()
        signal = df['x'].to_numpy()
        peaks = find_signal_peaks(signal, height, distance)
        plt.figure(figsize=(7,3))
        plt.plot(t, signal, label='x(t)')
        plt.plot(t[peaks], signal[peaks], 'ro', label='Peaks')
        plt.xlabel('Time (s)')
        plt.ylabel('Position x (cm)')
        plt.title(fname)
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_omega_vs_length(df_results: pd.DataFrame) -> None:
    markers = ['o', 's', '^']
    linestyles = ['-', '--', ':']
    plt.figure(figsize=(8,5))
    for (mass_label, group), marker, ls in zip(df_results.groupby('masa'), markers, linestyles):
        plt.errorbar(group['longitud_cm'], group['omega'], yerr=group['sigma_omega'],
                     fmt=marker, linestyle=ls, capsize=3, label=mass_label)
    plt.xlabel('Length (cm)')
    plt.ylabel('Angular frequency ω (rad/s)')
    plt.title('ω vs Length for different masses')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_omega_vs_mass(df_results: pd.DataFrame) -> None:
    plt.figure(figsize=(8,5))
    for length, group in df_results.groupby('longitud_cm'):
        plt.errorbar(group['masa_kg'], group['omega'], yerr=group['sigma_omega'],
                     fmt='o', capsize=3, label=f'L={length} cm')
    plt.xlabel('Mass (kg)')
    plt.ylabel('Angular frequency ω (rad/s)')
    plt.title('ω vs Mass for different lengths')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_omega_vs_amplitude(df_results: pd.DataFrame, mass_label: str, length_cm: float) -> None:
    group = df_results[(df_results['masa'] == mass_label) & (df_results['longitud_cm'] == length_cm)]
    plt.errorbar(group['amplitud_inicial'], group['omega'], yerr=group['sigma_omega'], fmt='o', capsize=3)
    plt.xlabel('Initial amplitude (cm)')
    plt.ylabel('Angular frequency ω (rad/s)')
    plt.title(f'ω vs Initial amplitude for mass={mass_label}, length={length_cm} cm')
    plt.grid(True)
    plt.show()

def regression_T2_vs_L(df_results: pd.DataFrame) -> pd.DataFrame:
    g_estimados = []
    for mass_label, group in df_results.groupby('masa'):
        L_m = group['longitud_cm'].to_numpy() / 100
        T2 = group['periodo'].to_numpy()**2
        sigma_T = group['sigma_periodo'].to_numpy()
        sigma_T2 = 2 * group['periodo'] * sigma_T
        slope, intercept, _, _, std_err = stats.linregress(L_m, T2)
        g_est = 4 * np.pi**2 / slope
        sigma_g = (4 * np.pi**2 / slope**2) * std_err

        g_estimados.append({
            'masa': mass_label,
            'g_estimado': g_est,
            'sigma_g': sigma_g,
            'pendiente': slope,
            'error_pendiente': std_err,
            'intercepto': intercept
        })

        plt.errorbar(L_m, T2, yerr=sigma_T2, fmt='o', capsize=3, label='Data')
        x_fit = np.linspace(L_m.min(), L_m.max(), 100)
        plt.plot(x_fit, slope*x_fit + intercept, 'r-', label=f'Fit slope={slope:.3f}')
        plt.xlabel('Length (m)')
        plt.ylabel('$T^2$ (s²)')
        plt.title(f'Regression T² vs L for mass {mass_label}\nEstimated g = {g_est:.3f} ± {sigma_g:.3f} m/s²')
        plt.legend()
        plt.grid(True)
        plt.show()

    return pd.DataFrame(g_estimados)

def export_results(df_results: pd.DataFrame, df_g: pd.DataFrame) -> None:
    df_results.to_csv('resultados_pendulo_con_incertidumbres.csv', index=False)
    df_g.to_csv('estimacion_g_por_masa.csv', index=False)
    logging.info("Saved resultados_pendulo_con_incertidumbres.csv and estimacion_g_por_masa.csv")

def main(data_folder: str, distance: int = 15) -> None:
    mass_dict_g = {'Madera': 5.1, 'Aluminio': 22.1, 'Bronce': 72.6}
    mass_dict_kg = {k: v/1000 for k, v in mass_dict_g.items()}

    filenames = sorted([f for f in os.listdir(data_folder) if f.endswith('.txt')])
    dfs = [load_data(os.path.join(data_folder, f)) for f in filenames]

    results = []

    for df, fname in zip(dfs, filenames):
        signal = df['x'].to_numpy()
        t = df['t'].to_numpy()
        height = calculate_height(signal)
        peaks = find_signal_peaks(signal, height, distance)
        omega, sigma_omega, period, sigma_period = calculate_omega_and_period(t, peaks)
        if np.isnan(omega):
            logging.warning(f"Not enough peaks in {fname}, skipping.")
            continue

        parts = fname.replace('.txt', '').split('_')
        length_cm = float(parts[0].replace('cm', ''))
        mass_label = parts[1]
        mass_kg = mass_dict_kg.get(mass_label, np.nan)

        initial_amp = get_initial_amplitude(signal, peaks)

        results.append({
            'archivo': fname,
            'masa': mass_label,
            'masa_kg': mass_kg,
            'longitud_cm': length_cm,
            'amplitud_inicial': initial_amp,
            'omega': omega,
            'sigma_omega': sigma_omega,
            'periodo': period,
            'sigma_periodo': sigma_period
        })

    df_results = pd.DataFrame(results)
    logging.info(f"Processed {len(df_results)} files successfully.")

    # Batch plots
    plot_omega_vs_length(df_results)
    plot_omega_vs_mass(df_results)

    if not df_results.empty:
        # Plot omega vs amplitude for first mass and length as example
        mass_example = df_results['masa'].iloc[0]
        length_example = df_results['longitud_cm'].iloc[0]
        plot_omega_vs_amplitude(df_results, mass_example, length_example)

    df_g = regression_T2_vs_L(df_results)
    export_results(df_results, df_g)

if __name__ == '__main__':
    data_folder = r'C:\Users\morph\Desktop\UDESA\fisica\Informes_Fisica\2do_Informe\CSVs' 
    main(data_folder)
