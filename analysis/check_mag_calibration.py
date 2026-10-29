import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repository root
from data_path import DATA_DIR  # noqa: E402
import numpy as np
import matplotlib.pyplot as plt
import os
import data_loader as dl


def analyze_and_plot_mag(dataset_dir):
    print(f"Loading data from {os.path.basename(dataset_dir)}...")
    t_imu, gyro, accel, mag = dl.load_sensor_data(dataset_dir)

    # דילול דגימות להצגה נקייה (לוקחים כל דגימה 10)
    mag_sub = mag[::10]
    mx, my, mz = mag_sub[:, 0], mag_sub[:, 1], mag_sub[:, 2]

    # --- חישוב מדדים מספריים לדו"ח ---
    center = np.mean(mag, axis=0)  # וקטור Offset (Hard-Iron)
    distances = np.linalg.norm(mag - center, axis=1)
    mean_radius = np.mean(distances)
    std_radius = np.std(distances)

    print("\n" + "=" * 50)
    print(f"   MAGNETOMETER ANALYSIS REPORT: {os.path.basename(dataset_dir)}")
    print("=" * 50)
    print(f"Estimated Hard-Iron Offset (b) : X={center[0]:.6f}, Y={center[1]:.6f}, Z={center[2]:.6f}")
    print(f"Mean Magnetic Field Radius (r) : {mean_radius:.6f}")
    print(f"Radius Variation (Soft-Iron/Noise) : Std={std_radius:.6f} ({std_radius / mean_radius * 100:.2f}%)")
    print("=" * 50 + "\n")

    # --- יצירת הגרף המורכב ---
    fig = plt.figure(figsize=(14, 10))

    # 1. גרף תלת מימדי
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.scatter(mx, my, mz, s=3, alpha=0.3, color='dodgerblue', label='Mag Data')
    ax3d.scatter([0], [0], [0], color='red', s=80, marker='X', label='Origin (0,0,0)')
    ax3d.scatter([center[0]], [center[1]], [center[2]], color='green', s=80, marker='o', label='Center of Mass')
    ax3d.set_title("3D Scatter Plot")
    ax3d.legend(loc='upper right', fontsize=8)

    # 2. היטל XY
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.scatter(mx, my, s=2, alpha=0.3, color='crimson')
    ax_xy.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax_xy.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax_xy.scatter(center[0], center[1], color='green', marker='o', s=50)
    ax_xy.set_title("Top View (XY Plane)")
    ax_xy.set_xlabel("Mag X")
    ax_xy.set_ylabel("Mag Y")
    ax_xy.grid(True)

    # 3. היטל XZ
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.scatter(mx, mz, s=2, alpha=0.3, color='purple')
    ax_xz.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax_xz.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax_xz.scatter(center[0], center[2], color='green', marker='o', s=50)
    ax_xz.set_title("Side View (XZ Plane)")
    ax_xz.set_xlabel("Mag X")
    ax_xz.set_ylabel("Mag Z")
    ax_xz.grid(True)

    # 4. היטל YZ
    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.scatter(my, mz, s=2, alpha=0.3, color='darkgreen')
    ax_yz.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax_yz.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax_yz.scatter(center[1], center[2], color='green', marker='o', s=50)
    ax_yz.set_title("Front View (YZ Plane)")
    ax_yz.set_xlabel("Mag Y")
    ax_yz.set_ylabel("Mag Z")
    ax_yz.grid(True)

    fig.suptitle(f"Magnetometer Distortion & Offset Analysis - {os.path.basename(dataset_dir)}", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    BASE_DIR = DATA_DIR
    flight_num=3
    dataset_path = os.path.join(BASE_DIR, f"mars_{flight_num}_sensors")
    analyze_and_plot_mag(dataset_path)