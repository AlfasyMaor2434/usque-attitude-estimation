import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repository root
from data_path import DATA_DIR  # noqa: E402
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import data_loader as dl


def batch_analyze_mag(base_dir, num_flights=19):
    print(f"Starting Batch Analysis for {num_flights} flights...")

    results = []
    all_mag_data = []  # לאיסוף כלל הנתונים לגרף ה-3D

    for i in range(1, num_flights + 1):
        dataset_dir = os.path.join(base_dir, f"mars_{i}_sensors")
        if not os.path.exists(dataset_dir):
            print(f"  [!] Skipped mars_{i} - Directory not found.")
            continue

        try:
            t_imu, gyro, accel, mag = dl.load_sensor_data(dataset_dir)
        except Exception as e:
            print(f"  [!] Error loading mars_{i}: {e}")
            continue

        # סינון אפסים ואינטרפולציות שגויות, והמרה ישירה ל-MicroTesla
        valid_mag = mag[np.linalg.norm(mag, axis=1) > 1e-6] * 1e6

        if len(valid_mag) == 0:
            print(f"  [!] Skipped mars_{i} - No valid magnetometer data.")
            continue

        # איסוף נקודות מדוללות לגרף המאוחד (אחת ל-50 דגימות כדי לא לחנוק את הזיכרון)
        all_mag_data.append(valid_mag[::50])

        # --- חישוב המדדים ---
        center = np.mean(valid_mag, axis=0)
        offset_mag = np.linalg.norm(center)

        distances = np.linalg.norm(valid_mag - center, axis=1)
        mean_r = np.mean(distances)
        std_r = np.std(distances)
        min_r = np.min(distances)
        max_r = np.max(distances)
        var_pct = (std_r / mean_r * 100) if mean_r > 0 else 0

        # שמירה במילון
        results.append({
            'Flight': f"mars_{i}",
            'Offset_X_uT': center[0],
            'Offset_Y_uT': center[1],
            'Offset_Z_uT': center[2],
            'Offset_Magnitude_uT': offset_mag,
            'Mean_Radius_uT': mean_r,
            'Radius_Std_uT': std_r,
            'Variation_Percent': var_pct,
            'Min_Radius_uT': min_r,
            'Max_Radius_uT': max_r
        })
        print(f"  [v] Processed mars_{i} (Mean R: {mean_r:.2f} uT, Offset: {offset_mag:.2f} uT)")

    if not results:
        print("No data processed.")
        return

    # --- חישוב שורת הממוצע הכולל ---
    avg_row = {'Flight': 'AVERAGE'}
    keys_to_avg = list(results[0].keys())[1:]  # כל המפתחות פרט לשם הטיסה
    for k in keys_to_avg:
        avg_row[k] = np.mean([row[k] for row in results])

    # --- כתיבה לקובץ CSV ---
    csv_filename = "mag_batch_analysis.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            # עיגול נתונים לכתיבה נקייה יותר לקובץ
            rounded_row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            writer.writerow(rounded_row)

        # כתיבת שורת הממוצע בסוף
        rounded_avg = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in avg_row.items()}
        writer.writerow(rounded_avg)

    print(f"\n>>> Results successfully saved to {csv_filename} <<<\n")

    # --- ציור הגרפים ---
    plot_batch_results(results, np.vstack(all_mag_data))


def plot_batch_results(results, combined_mag_points):
    flights = [r['Flight'] for r in results]
    off_x = [r['Offset_X_uT'] for r in results]
    off_y = [r['Offset_Y_uT'] for r in results]
    off_z = [r['Offset_Z_uT'] for r in results]

    mean_r = [r['Mean_Radius_uT'] for r in results]
    std_r = [r['Radius_Std_uT'] for r in results]

    x_indices = np.arange(len(flights))

    # ==========================================
    # Figure 1: ניתוח סטטיסטי דו-מימדי (Hard & Soft Iron)
    # ==========================================
    fig1 = plt.figure(figsize=(16, 6))

    # --- גרף 1: רכיבי ה-Hard-Iron Offset לאורך הטיסות ---
    ax1 = fig1.add_subplot(1, 2, 1)
    width = 0.25
    ax1.bar(x_indices - width, off_x, width, label='Offset X', color='crimson')
    ax1.bar(x_indices, off_y, width, label='Offset Y', color='dodgerblue')
    ax1.bar(x_indices + width, off_z, width, label='Offset Z', color='forestgreen')

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(flights, rotation=45, ha='right')
    ax1.set_ylabel('Hard-Iron Offset (uT)')
    ax1.set_title('Consistency of Hard-Iron Offset Across Flights')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- גרף 2: רדיוס השדה (Soft-Iron / רעש) ---
    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.errorbar(x_indices, mean_r, yerr=std_r, fmt='o', color='purple',
                 capsize=5, capthick=2, markerfacecolor='black', markersize=6, label='Mean Radius ± Std')

    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(flights, rotation=45, ha='right')
    ax2.set_ylabel('Magnetic Field Radius (uT)')
    ax2.set_title('Field Radius and Variation (Soft-Iron Effect)')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    fig1.tight_layout()

    # ==========================================
    # Figure 2: תצוגה תלת-מימדית של כלל הנתונים
    # ==========================================
    fig2 = plt.figure(figsize=(10, 8))
    ax3 = fig2.add_subplot(1, 1, 1, projection='3d')
    global_center = np.mean(combined_mag_points, axis=0)

    # תצוגה של כל הנקודות במרחב
    ax3.scatter(combined_mag_points[:, 0], combined_mag_points[:, 1], combined_mag_points[:, 2],
                s=1, alpha=0.1, color='teal', label='Aggregated Data')
    ax3.scatter(global_center[0], global_center[1], global_center[2],
                color='red', s=100, marker='X', label='Global Offset Center')

    # קיבוע צירים כדי למנוע מתיחה של התצוגה (axis equal)
    max_range = np.array([combined_mag_points[:, 0].max() - combined_mag_points[:, 0].min(),
                          combined_mag_points[:, 1].max() - combined_mag_points[:, 1].min(),
                          combined_mag_points[:, 2].max() - combined_mag_points[:, 2].min()]).max() / 2.0
    mid_x = (combined_mag_points[:, 0].max() + combined_mag_points[:, 0].min()) * 0.5
    mid_y = (combined_mag_points[:, 1].max() + combined_mag_points[:, 1].min()) * 0.5
    mid_z = (combined_mag_points[:, 2].max() + combined_mag_points[:, 2].min()) * 0.5
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)

    ax3.set_xlabel('Mag X (uT)')
    ax3.set_ylabel('Mag Y (uT)')
    ax3.set_zlabel('Mag Z (uT)')
    ax3.set_title(f'Grand Unified Magnetometer Scatter ({len(flights)} Flights Combined)')
    ax3.legend()

    fig2.tight_layout()

    # הצגת כל החלונות יחד
    plt.show()


if __name__ == "__main__":
    BASE_DIR = DATA_DIR
    # מריץ את כל הטיסות מ-1 עד 19
    batch_analyze_mag(BASE_DIR, num_flights=19)