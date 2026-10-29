"""
FFT.py
------
Analyzes the frequency spectrum of IMU sensors (PX4 and LSM9DS1).
Helps in identifying motor vibration frequencies and selecting the optimal Low-Pass Filter cutoff.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repository root
from data_path import DATA_DIR  # noqa: E402

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ==========================================
# CONFIGURATION PANEL
# ==========================================
BASE_DIR = DATA_DIR

# Choose which dataset type to analyze: 'MARS' or 'VIBRATION'
DATASET_MODE = 'VIBRATION'

# If 'MARS', choose the flight number (1 to 19)
MARS_FLIGHT_NUM = 4

# If 'VIBRATION', choose the folder name (e.g., 'v10', 'v20', 'v40', 'v60', 'v80', 'v100')
VIBRATION_LEVEL = 'v100'


# ==========================================

def get_dataset_path():
    if DATASET_MODE == 'MARS':
        return os.path.join(BASE_DIR, f"mars_{MARS_FLIGHT_NUM}_sensors")
    elif DATASET_MODE == 'VIBRATION':
        return os.path.join(BASE_DIR, "insane_vibration_analysis_sensor_data", VIBRATION_LEVEL)
    else:
        raise ValueError("Invalid DATASET_MODE. Use 'MARS' or 'VIBRATION'.")


def load_imu_csv(file_path):
    """ Loads standard IMU CSV files (t, ax, ay, az, wx, wy, wz) """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None, None, None

    try:
        data = np.loadtxt(file_path, delimiter=',', comments='#')
    except ValueError:
        data = np.loadtxt(file_path, delimiter=',', comments='#', skiprows=1)

    t = data[:, 0]
    accel = data[:, 1:4]
    gyro = data[:, 4:7]
    return t, accel, gyro


def compute_fft(t, data):
    """ Computes the Single-Sided Amplitude Spectrum using FFT """
    N = len(t)
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt

    # Detrend data (remove mean to remove the massive DC spike at 0Hz)
    data_detrended = data - np.mean(data, axis=0)

    # Compute FFT
    yf = rfft(data_detrended, axis=0)
    xf = rfftfreq(N, dt)

    # Calculate Magnitude
    magnitude = np.abs(yf) / N
    # Multiply by 2 (except DC and Nyquist) for single-sided spectrum
    magnitude[1:-1] *= 2

    return xf, magnitude, fs


def plot_sensor_fft(ax_row, xf, mag_accel, mag_gyro, fs, title_prefix):
    """ Helper to plot Accel and Gyro FFT on a given row of axes """
    axes = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    # Plot Accelerometer FFT
    for i in range(3):
        ax_row[0].plot(xf, mag_accel[:, i], label=f'Accel {axes[i]}', color=colors[i], alpha=0.8)

    ax_row[0].set_title(f"{title_prefix} - Accelerometer FFT (Sample Rate: {fs:.1f}Hz)")
    ax_row[0].set_ylabel("Amplitude $(m/s^2)$")
    ax_row[0].set_xlabel("Frequency (Hz)")
    ax_row[0].grid(True)
    ax_row[0].legend()
    # Limit X axis slightly below Nyquist for better view
    ax_row[0].set_xlim(0, fs / 2.0)

    # Plot Gyroscope FFT
    for i in range(3):
        ax_row[1].plot(xf, mag_gyro[:, i], label=f'Gyro {axes[i]}', color=colors[i], alpha=0.8)

    ax_row[1].set_title(f"{title_prefix} - Gyroscope FFT")
    ax_row[1].set_ylabel("Amplitude $(rad/s)$")
    ax_row[1].set_xlabel("Frequency (Hz)")
    ax_row[1].grid(True)
    ax_row[1].legend()
    ax_row[1].set_xlim(0, fs / 2.0)


def main():
    dataset_dir = get_dataset_path()
    print(f"Analyzing Dataset: {dataset_dir}")

    px4_file = os.path.join(dataset_dir, "px4_imu.csv")
    lsm_file = os.path.join(dataset_dir, "lsm_imu.csv")

    t_px4, acc_px4, gyr_px4 = load_imu_csv(px4_file)
    t_lsm, acc_lsm, gyr_lsm = load_imu_csv(lsm_file)

    if t_px4 is None and t_lsm is None:
        print("Error: Could not load any IMU data. Please check your path.")
        return

    # Setup Figure (2 rows: PX4, LSM | 2 cols: Accel, Gyro)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False)
    fig.suptitle(f"Frequency Spectrum Analysis (FFT)\nDataset: {os.path.basename(dataset_dir)}", fontsize=16)

    # Process and Plot PX4 (Typically ~200Hz)
    if t_px4 is not None:
        print("Computing FFT for PX4 IMU...")
        xf_px4_a, mag_acc_px4, fs_px4 = compute_fft(t_px4, acc_px4)
        xf_px4_g, mag_gyr_px4, _ = compute_fft(t_px4, gyr_px4)
        plot_sensor_fft(axes[0, :], xf_px4_a, mag_acc_px4, mag_gyr_px4, fs_px4, "PX4 IMU")
    else:
        axes[0, 0].text(0.5, 0.5, 'PX4 Data Not Found', ha='center')
        axes[0, 1].text(0.5, 0.5, 'PX4 Data Not Found', ha='center')

    # Process and Plot LSM (Typically ~800Hz)
    if t_lsm is not None:
        print("Computing FFT for LSM9DS1 IMU...")
        xf_lsm_a, mag_acc_lsm, fs_lsm = compute_fft(t_lsm, acc_lsm)
        xf_lsm_g, mag_gyr_lsm, _ = compute_fft(t_lsm, gyr_lsm)
        plot_sensor_fft(axes[1, :], xf_lsm_a, mag_acc_lsm, mag_gyr_lsm, fs_lsm, "LSM9DS1 IMU")
    else:
        axes[1, 0].text(0.5, 0.5, 'LSM Data Not Found', ha='center')
        axes[1, 1].text(0.5, 0.5, 'LSM Data Not Found', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()